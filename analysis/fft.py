"""
FFT analysis utilities for the communication system simulator.

This module calculates frequency-domain representations of signals.
It supports:
- single-sided FFT amplitude spectra
- dB conversion
- dominant frequency extraction
"""

import numpy as np


def compute_single_sided_fft(
    signal: np.ndarray,
    sample_rate: float,
    apply_window: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the single-sided amplitude spectrum of a real-valued signal.

    signal: input time-domain signal
    sample_rate: sampling rate in Hz
    apply_window: whether to apply a Hann window before FFT

    Returns:
    frequencies, magnitudes
    """
    if signal.size == 0:
        raise ValueError("signal must not be empty.")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")

    number_of_samples = signal.size

    # Remove DC offset so the zero-frequency component does not dominate the plot.
    centered_signal = signal - np.mean(signal)

    if apply_window:
        window = np.hanning(number_of_samples)
        windowed_signal = centered_signal * window

        # Coherent gain correction keeps amplitudes closer to their true values.
        coherent_gain = np.sum(window) / number_of_samples
    else:
        windowed_signal = centered_signal
        coherent_gain = 1.0

    fft_values = np.fft.rfft(windowed_signal)

    frequencies = np.fft.rfftfreq(
        n=number_of_samples,
        d=1 / sample_rate
    )

    magnitudes = np.abs(fft_values) / (number_of_samples * coherent_gain)

    # Convert from two-sided amplitude to single-sided amplitude.
    # Do not double DC or Nyquist components.
    if number_of_samples > 1:
        magnitudes[1:-1] *= 2

    return frequencies, magnitudes


def amplitude_to_db(
    magnitudes: np.ndarray,
    floor: float = 1e-12
) -> np.ndarray:
    """
    Convert amplitude magnitudes to decibels.

    Uses:
    magnitude_dB = 20 * log10(magnitude)

    floor prevents log10(0).
    """
    safe_magnitudes = np.maximum(magnitudes, floor)

    return 20 * np.log10(safe_magnitudes)


def find_dominant_frequencies(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    number_of_peaks: int = 5,
    minimum_frequency: float = 0.0
) -> list[tuple[float, float]]:
    """
    Find the largest frequency components in a spectrum.

    frequencies: frequency array in Hz
    magnitudes: magnitude array
    number_of_peaks: number of dominant frequencies to return
    minimum_frequency: ignore frequencies below this value

    Returns:
    list of (frequency, magnitude) pairs
    """
    if frequencies.shape != magnitudes.shape:
        raise ValueError("frequencies and magnitudes must have the same shape.")

    if number_of_peaks <= 0:
        raise ValueError("number_of_peaks must be greater than 0.")

    valid_indices = frequencies >= minimum_frequency

    valid_frequencies = frequencies[valid_indices]
    valid_magnitudes = magnitudes[valid_indices]

    sorted_indices = np.argsort(valid_magnitudes)[::-1]

    dominant = []

    for index in sorted_indices[:number_of_peaks]:
        dominant.append(
            (
                float(valid_frequencies[index]),
                float(valid_magnitudes[index])
            )
        )

    return dominant


def test_fft():
    """
    Simple import test.
    """
    return "analysis.fft imported correctly"