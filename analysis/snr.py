"""
SNR and error metric utilities for the communication system simulator.

This module calculates:
- signal power
- noise power
- measured SNR
- mean squared error
- root mean squared error
- normalized mean squared error
"""

import numpy as np


def calculate_power(signal: np.ndarray) -> float:
    """
    Calculate average power of a sampled signal.

    For a voltage-like signal:
    power = mean(signal^2)
    """
    if signal.size == 0:
        raise ValueError("signal must not be empty.")

    return float(np.mean(signal ** 2))


def calculate_snr_from_noise(
    clean_signal: np.ndarray,
    noise: np.ndarray
) -> float:
    """
    Calculate SNR in dB when the clean signal and noise are known.

    SNR_dB = 10 * log10(signal_power / noise_power)
    """
    if clean_signal.shape != noise.shape:
        raise ValueError("clean_signal and noise must have the same shape.")

    signal_power = calculate_power(clean_signal)
    noise_power = calculate_power(noise)

    if noise_power == 0:
        return float("inf")

    snr_db = 10 * np.log10(signal_power / noise_power)

    return float(snr_db)


def calculate_snr_db(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray
) -> float:
    """
    Calculate measured SNR in dB from a clean signal and noisy signal.

    noise = noisy_signal - clean_signal
    """
    if clean_signal.shape != noisy_signal.shape:
        raise ValueError("clean_signal and noisy_signal must have the same shape.")

    noise = noisy_signal - clean_signal

    return calculate_snr_from_noise(
        clean_signal=clean_signal,
        noise=noise
    )


def mean_squared_error(
    reference_signal: np.ndarray,
    test_signal: np.ndarray
) -> float:
    """
    Calculate mean squared error between a reference signal and test signal.

    MSE = mean((reference - test)^2)
    """
    if reference_signal.shape != test_signal.shape:
        raise ValueError("reference_signal and test_signal must have the same shape.")

    error = reference_signal - test_signal

    return float(np.mean(error ** 2))


def root_mean_squared_error(
    reference_signal: np.ndarray,
    test_signal: np.ndarray
) -> float:
    """
    Calculate root mean squared error.

    RMSE = sqrt(MSE)
    """
    mse = mean_squared_error(
        reference_signal=reference_signal,
        test_signal=test_signal
    )

    return float(np.sqrt(mse))


def normalized_mean_squared_error(
    reference_signal: np.ndarray,
    test_signal: np.ndarray
) -> float:
    """
    Calculate normalized mean squared error.

    NMSE = MSE / reference_signal_power

    This gives an error value relative to the power of the original signal.
    """
    mse = mean_squared_error(
        reference_signal=reference_signal,
        test_signal=test_signal
    )

    reference_power = calculate_power(reference_signal)

    if reference_power == 0:
        raise ValueError("reference signal power must not be zero.")

    return float(mse / reference_power)


def test_snr():
    """
    Simple import test.
    """
    return "analysis.snr imported correctly"