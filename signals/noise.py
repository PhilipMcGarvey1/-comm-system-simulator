"""
Noise utilities for the communication system simulator.

This module adds channel noise to transmitted signals.
The main channel model used here is AWGN:
Additive White Gaussian Noise.
"""

import numpy as np


def calculate_signal_power(signal: np.ndarray) -> float:
    """
    Calculate average signal power.

    For a sampled signal, average power is approximated as:

    P = mean(signal^2)

    This assumes the signal is represented as voltage-like samples
    across time.
    """
    if signal.size == 0:
        raise ValueError("signal must not be empty.")

    return float(np.mean(signal ** 2))


def generate_awgn(
    signal: np.ndarray,
    snr_db: float,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate Additive White Gaussian Noise for a desired SNR.

    signal: clean input signal
    snr_db: desired signal-to-noise ratio in dB
    seed: optional random seed for repeatable results

    Returns:
    noise array with the same shape as the input signal
    """
    signal_power = calculate_signal_power(signal)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise_std = np.sqrt(noise_power)

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=signal.shape
    )

    return noise


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add AWGN to a clean signal.

    Returns:
    noisy_signal, noise
    """
    noise = generate_awgn(
        signal=signal,
        snr_db=snr_db,
        seed=seed
    )

    noisy_signal = signal + noise

    return noisy_signal, noise


def test_noise():
    """
    Simple import test.
    """
    return "signals.noise imported correctly"