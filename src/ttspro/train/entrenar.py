"""VITS training loop for `ttspro.model.Sintetizador`.

    uv run python -m ttspro.train.entrenar --cache cache/x --salida runs/x --batch 16 --fp16

One GPU (or CPU, slowly — the decided fallback). `paso()` is a pure function
of (models, optimizers, batch) so a test can run it on random data; the
script around it does data, logging, checkpoints and listening samples.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ttspro.data.dataset import DatasetTTS, LotesPorLongitud, collate
from ttspro.model.comun import slice_segments
from ttspro.model.config import ConfigSintetizador
from ttspro.model.sintetizador import Sintetizador
from ttspro.train.discriminador import MultiPeriodDiscriminator
from ttspro.train.mel import mel_spectrogram_torch, spec_to_mel_torch
from ttspro.train.perdidas import discriminator_loss, feature_loss, generator_loss, kl_loss

RAIZ = Path(__file__).resolve().parents[3]
C_MEL = 45.0
C_KL = 1.0


def config_por_defecto() -> ConfigSintetizador:
    from ttspro.export.tts import config_por_defecto as _c

    return _c()


def paso(
    net_g,
    net_d,
    optim_g,
    optim_d,
    scaler,
    lote: dict,
    cfg: ConfigSintetizador,
    device,
    fp16: bool,
    solo_disc: bool = False,
    scl: nn.Module | None = None,
    peso_scl: float = 0.0,
) -> dict[str, float]:
    """One VITS step. `solo_disc` updates the discriminator only (warm-up when the
    generator comes pre-trained and the discriminator does not: a fresh critic
    feeds a good generator noise for its first steps)."""
    tokens = lote["tokens"].to(device)
    tokens_len = lote["tokens_len"].to(device)
    spec = lote["spec"].to(device)
    spec_len = lote["spec_len"].to(device)
    onda = lote["onda"].to(device)
    emb = lote["embedding"].to(device)
    idioma = lote["idioma"].to(device)
    tipo_autocast = "cuda" if device.type == "cuda" else "cpu"

    with torch.autocast(
        tipo_autocast,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=fp16,
    ):
        y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (
            net_g(tokens, tokens_len, spec, spec_len, emb, idioma)
        )
        mel = spec_to_mel_torch(
            spec.float(), cfg.n_fft, cfg.n_mels, cfg.sample_rate, cfg.mel_fmin, cfg.mel_fmax
        )
        y_mel = slice_segments(mel, ids_slice, cfg.segment_size // cfg.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            cfg.n_fft,
            cfg.n_mels,
            cfg.sample_rate,
            cfg.hop_length,
            cfg.win_length,
            cfg.mel_fmin,
            cfg.mel_fmax,
        )
        y = slice_segments(onda, ids_slice * cfg.hop_length, cfg.segment_size)
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
    optim_d.zero_grad(set_to_none=True)
    scaler.scale(loss_disc).backward()
    scaler.unscale_(optim_d)
    grad_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), 1e9)
    scaler.step(optim_d)

    with torch.autocast(
        tipo_autocast,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=fp16,
    ):
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        loss_dur = torch.sum(l_length.float())
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * C_MEL
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * C_KL
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    loss_scl = torch.zeros((), device=device)
    if scl is not None and peso_scl > 0:
        loss_scl = scl(y_hat, y) * peso_scl  # fp32 inside (the fbank overflows fp16)
        loss_gen_all = loss_gen_all + loss_scl
    optim_g.zero_grad(set_to_none=True)
    if solo_disc:
        grad_g = torch.zeros(())
    else:
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1e9)
        scaler.step(optim_g)
    scaler.update()
    return {
        "disc": float(loss_disc),
        "gen": float(loss_gen),
        "fm": float(loss_fm),
        "mel": float(loss_mel),
        "dur": float(loss_dur),
        "kl": float(loss_kl),
        "scl": float(loss_scl),
        "total_g": float(loss_gen_all),
        "grad_g": float(grad_g),
        "grad_d": float(grad_d),
    }


def construir(cfg: ConfigSintetizador, device, lr: float):
    net_g = Sintetizador(cfg).to(device)
    net_d = MultiPeriodDiscriminator().to(device)
    optim_g = torch.optim.AdamW(net_g.parameters(), lr, betas=(0.8, 0.99), eps=1e-9)
    optim_d = torch.optim.AdamW(net_d.parameters(), lr, betas=(0.8, 0.99), eps=1e-9)
    return net_g, net_d, optim_g, optim_d


def guardar(
    ruta: Path, net_g, net_d, optim_g, optim_d, paso_n: int, epoca: int, cfg: ConfigSintetizador
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "modelo": net_g.state_dict(),
            "disc": net_d.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "paso": paso_n,
            "epoca": epoca,
            "cfg": dataclasses.asdict(cfg),
        },
        ruta,
    )


def muestras(
    net_g,
    cfg: ConfigSintetizador,
    indice: list[dict],
    carpeta: Path,
    paso_n: int,
    device,
    n: int = 3,
) -> None:
    """Synthesize a few training sentences with their own speaker embedding: listening material."""
    import soundfile as sf

    from ttspro.data.dataset import DatasetTTS

    carpeta.mkdir(parents=True, exist_ok=True)
    idiomas = DatasetTTS([], cfg).idiomas
    net_g.eval()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for i, e in enumerate(indice[:n]):
            tokens = e["tokens"].unsqueeze(0).to(device)
            longitud = tokens.shape[1]
            onda = net_g.inferir(
                tokens,
                torch.tensor([longitud], device=device),
                e["embedding"].unsqueeze(0).to(device),
                torch.tensor([idiomas[e["idioma"]]], device=device),
                torch.randn(1, cfg.inter_channels, longitud * 12, generator=g).to(device),
                torch.randn(1, 2, longitud, generator=g).to(device),
                torch.tensor([0.667, 0.8, 1.0], device=device),
            )
            sf.write(
                str(carpeta / f"paso{paso_n:07d}_{i}_{e['idioma']}.wav"),
                onda[0, 0].cpu().numpy(),
                cfg.sample_rate,
            )
    net_g.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache",
        type=Path,
        required=True,
        nargs="+",
        help="carpeta(s) con indice.pt de ttspro.data.preparar; varias se concatenan (es + en)",
    )
    ap.add_argument(
        "--calentar-disc",
        type=int,
        default=0,
        help="pasos iniciales solo del discriminador (generador congelado); util con --init",
    )
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epocas", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cada-log", type=int, default=50)
    ap.add_argument("--cada-checkpoint", type=int, default=2000)
    ap.add_argument("--reanudar", type=Path, default=None)
    ap.add_argument(
        "--init",
        type=Path,
        default=None,
        help="pesos iniciales del generador (ttspro.train.inicializar); optimizadores desde cero",
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--scl",
        type=float,
        default=0.0,
        help="peso de la pérdida de consistencia de locutor (YourTTS usa 9); 0 la apaga",
    )
    ap.add_argument("--scl-desde", type=int, default=0, help="paso en el que empieza la SCL")
    ap.add_argument(
        "--balancear",
        action="store_true",
        help="recorta cada idioma al tamaño del más pequeño; sin esto el inglés "
        "(VCTK + LibriTTS-R) triplica al español y el modelo aprende sobre todo inglés",
    )
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    cfg = config_por_defecto()
    inicial = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    if inicial is not None and "cfg" in inicial:
        cfg = ConfigSintetizador(**inicial["cfg"])  # the init decides the architecture
    torch.manual_seed(0)
    net_g, net_d, optim_g, optim_d = construir(cfg, device, args.lr)
    if inicial is not None:
        net_g.load_state_dict(inicial["modelo"])
        print(f"generador inicializado desde {args.init}")
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999875)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999875)
    paso_n, epoca0 = 0, 0
    if args.reanudar:
        estado = torch.load(args.reanudar, map_location=device)
        net_g.load_state_dict(estado["modelo"])
        net_d.load_state_dict(estado["disc"])
        optim_g.load_state_dict(estado["optim_g"])
        optim_d.load_state_dict(estado["optim_d"])
        paso_n, epoca0 = estado["paso"], estado["epoca"]

    consistencia = None
    if args.scl > 0:
        from ttspro.train.consistencia import ConsistenciaLocutor

        consistencia = ConsistenciaLocutor().to(device)
        print(f"consistencia de locutor activa: peso {args.scl} desde el paso {args.scl_desde}")

    indice = [e for carpeta in args.cache for e in torch.load(carpeta / "indice.pt")]
    conteo = Counter(e["idioma"] for e in indice)
    if args.balancear and len(conteo) > 1:
        tope = min(conteo.values())
        rng = random.Random(0)
        por_idioma: dict[str, list] = {}
        for e in indice:
            por_idioma.setdefault(e["idioma"], []).append(e)
        indice = []
        for _, ejemplos in sorted(por_idioma.items()):
            rng.shuffle(ejemplos)
            indice.extend(ejemplos[:tope])
        print(f"balanceado a {tope} frases por idioma (de {dict(conteo)})")
    print(f"idiomas: {dict(Counter(e['idioma'] for e in indice))}")
    dataset = DatasetTTS(indice, cfg)
    loader = DataLoader(
        dataset,
        batch_sampler=LotesPorLongitud(dataset, args.batch),
        collate_fn=collate,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    args.salida.mkdir(parents=True, exist_ok=True)
    registro = (args.salida / "registro.jsonl").open("a", encoding="utf-8")
    print(
        f"{len(dataset)} frases, {len(loader)} lotes/época, "
        f"device={device}, fp16={scaler.is_enabled()}"
    )

    t0 = time.time()
    for epoca in range(epoca0, args.epocas):
        net_g.train()
        net_d.train()
        for lote in loader:
            metricas = paso(
                net_g,
                net_d,
                optim_g,
                optim_d,
                scaler,
                lote,
                cfg,
                device,
                scaler.is_enabled(),
                solo_disc=paso_n < args.calentar_disc,
                scl=consistencia,
                peso_scl=args.scl if paso_n >= args.scl_desde else 0.0,
            )
            paso_n += 1
            if paso_n % args.cada_log == 0:
                fila = {
                    "paso": paso_n,
                    "epoca": epoca,
                    "s": round(time.time() - t0),
                    "lr": sched_g.get_last_lr()[0],
                    **{k: round(v, 4) for k, v in metricas.items()},
                }
                registro.write(json.dumps(fila) + "\n")
                registro.flush()
                print(fila)
            if paso_n % args.cada_checkpoint == 0:
                guardar(
                    args.salida / f"G_{paso_n}.pt",
                    net_g,
                    net_d,
                    optim_g,
                    optim_d,
                    paso_n,
                    epoca,
                    cfg,
                )
                muestras(net_g, cfg, indice, args.salida / "muestras", paso_n, device)
        sched_g.step()
        sched_d.step()
    guardar(
        args.salida / f"G_{paso_n}.pt", net_g, net_d, optim_g, optim_d, paso_n, args.epocas, cfg
    )


if __name__ == "__main__":
    main()
