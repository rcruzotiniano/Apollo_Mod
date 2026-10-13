from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

MIN_CUTOFF_HZ = 1000
MAX_CUTOFF_HZ = 24000


class CutoffError(ValueError):
    """El cutoff pedido no es aplicable."""


def normalize_cutoff(cutoff_hz) -> int:
    """Valida el cutoff como entero y lo devuelve normalizado."""
    try:
        return int(cutoff_hz)
    except (TypeError, ValueError):
        raise CutoffError(f"El cutoff debe ser un numero entero, no {cutoff_hz!r}.") from None


def validate_cutoff(cutoff_hz: int, sample_rate: int) -> int:
    """Comprueba rango y Nyquist. Devuelve el cutoff normalizado."""
    cutoff_hz = normalize_cutoff(cutoff_hz)
    if cutoff_hz == 0:
        return 0
    if not MIN_CUTOFF_HZ <= cutoff_hz <= MAX_CUTOFF_HZ:
        raise CutoffError(
            f"El cutoff debe ser 0 (sin cutoff) o un valor de "
            f"{MIN_CUTOFF_HZ} a {MAX_CUTOFF_HZ} Hz; se recibio {cutoff_hz} Hz."
        )
    if cutoff_hz >= sample_rate / 2:
        raise CutoffError(
            f"El cutoff ({cutoff_hz} Hz) debe quedar por debajo de Nyquist "
            f"({sample_rate / 2:g} Hz) para una frecuencia de muestreo de "
            f"{sample_rate} Hz."
        )
    return cutoff_hz


def apply_input_cutoff(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: int,
) -> np.ndarray:
    """Aplica un pasabajos de fase cero al audio de entrada.

    El cutoff es libre: cualquier valor entre 1000 y 24000 Hz, por debajo de
    Nyquist. ``0`` (o desactivado) devuelve el audio intacto.
    """
    cutoff_hz = validate_cutoff(cutoff_hz, sample_rate)
    if cutoff_hz == 0:
        return audio
    if audio.shape[-1] == 0:
        raise ValueError("No se puede filtrar un archivo de audio vacío.")

    sections = butter(
        8,
        cutoff_hz,
        btype="lowpass",
        fs=sample_rate,
        output="sos",
    )
    # SciPy 1.13+ exige sos en float64; librosa devuelve float32.
    sections = np.asarray(sections, dtype=np.float64)
    mono = audio.ndim == 1
    signal = audio.reshape(1, -1) if mono else audio
    pad_length = min(3 * (2 * len(sections) + 1), signal.shape[-1] - 1)
    filtered = sosfiltfilt(
        sections,
        signal.astype(np.float64, copy=False),
        axis=-1,
        padlen=pad_length,
    )
    return filtered.astype(audio.dtype, copy=False).reshape(audio.shape)
