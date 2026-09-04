"""VITS losses."""

from __future__ import annotations

import torch


def feature_loss(fmap_r, fmap_g) -> torch.Tensor:
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g, strict=True):
        for rl, gl in zip(dr, dg, strict=True):
            loss = loss + torch.mean(torch.abs(rl.float().detach() - gl.float()))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs, strict=True):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss = loss + r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l_ = torch.mean((1 - dg) ** 2)
        gen_losses.append(l_)
        loss = loss + l_
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask) -> torch.Tensor:
    """KL(q(z|y) || p(z|x)) over masked frames."""
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()
    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask)
