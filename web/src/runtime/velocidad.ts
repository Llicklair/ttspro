/**
 * Cambiar la velocidad del habla **sin tocar el tono**.
 *
 * Pocket TTS no tiene control de velocidad: decodifica a 12,5 marcos por segundo
 * y no acepta un factor. Lo obvio sería subir el `playbackRate` del nodo de
 * audio, pero eso reproduce más deprisa *y más agudo* — y en una voz clonada eso
 * es exactamente lo que no se puede tocar, porque el tono es la mitad de lo que
 * la hace reconocible.
 *
 * Así que se estira el tiempo: WSOLA (waveform similarity overlap-add). Se corta
 * la onda en trozos solapados, y antes de pegar cada uno se busca, dentro de una
 * ventana pequeña, el desplazamiento en el que **mejor encaja** con lo que ya va
 * escrito. Encajar por correlación en vez de pegar a ciegas es lo que evita los
 * clics y el timbre metálico del solapamiento simple, y el tono no se mueve
 * porque los trozos nunca se remuestrean.
 *
 * Supertonic sí tiene `velocidad` dentro del modelo y no necesita esto: allí la
 * duración la predice la red, que es mejor que estirar después.
 */

/** Ventana de análisis y salto, en muestras a 24 kHz. Escalados si la frecuencia difiere. */
const VENTANA_BASE = 1024;
const SALTO_BASE = 256;
/** Cuánto se permite moverse a un trozo para encajar mejor. */
const BUSQUEDA_BASE = 256;

/**
 * `factor` > 1 habla más rápido, < 1 más despacio. Fuera de [0,5, 2] no merece la
 * pena: el resultado deja de parecer una persona hablando.
 */
export function cambiarVelocidad(
  onda: Float32Array,
  factor: number,
  sampleRate = 24000,
): Float32Array {
  if (!Number.isFinite(factor) || Math.abs(factor - 1) < 0.01 || onda.length === 0) return onda;
  const f = Math.min(2, Math.max(0.5, factor));

  const escala = sampleRate / 24000;
  const ventana = Math.round(VENTANA_BASE * escala);
  const saltoSalida = Math.round(SALTO_BASE * escala);
  const busqueda = Math.round(BUSQUEDA_BASE * escala);
  // Leer más rápido de lo que se escribe es hablar más rápido.
  const saltoEntrada = Math.round(saltoSalida * f);
  if (onda.length < ventana * 2) return onda;

  const hann = new Float32Array(ventana);
  for (let i = 0; i < ventana; i++) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / ventana);

  const largo = Math.ceil(onda.length / f) + ventana;
  const salida = new Float32Array(largo);
  const pesos = new Float32Array(largo);

  let leer = 0;
  let escribir = 0;
  while (leer + ventana + busqueda < onda.length && escribir + ventana < largo) {
    // Dónde encaja mejor este trozo con lo que ya hay escrito. En el primero no
    // hay nada con lo que comparar, así que se pega tal cual.
    let mejor = 0;
    if (escribir > 0) {
      let mejorPuntuacion = Number.NEGATIVE_INFINITY;
      const solape = Math.min(saltoSalida, ventana);
      for (let d = -busqueda; d <= busqueda; d++) {
        const desde = leer + d;
        if (desde < 0 || desde + solape >= onda.length) continue;
        let suma = 0;
        // Un paso de 4 muestras: la correlación no cambia de forma por mirar
        // una de cada cuatro, y esto corre en el hilo de la página.
        for (let i = 0; i < solape; i += 4) suma += onda[desde + i] * salida[escribir + i];
        if (suma > mejorPuntuacion) {
          mejorPuntuacion = suma;
          mejor = d;
        }
      }
    }
    const desde = Math.max(0, Math.min(onda.length - ventana, leer + mejor));
    for (let i = 0; i < ventana; i++) {
      salida[escribir + i] += onda[desde + i] * hann[i];
      pesos[escribir + i] += hann[i];
    }
    leer += saltoEntrada;
    escribir += saltoSalida;
  }

  // Deshacer la ventana: donde se han solapado dos trozos, la suma de las Hann
  // no es 1 y sin esto la onda sale con ondulaciones de volumen.
  const fin = Math.min(escribir + ventana, largo);
  const recortada = salida.subarray(0, fin);
  for (let i = 0; i < fin; i++) {
    if (pesos[i] > 1e-6) recortada[i] /= pesos[i];
  }
  return recortada.slice();
}
