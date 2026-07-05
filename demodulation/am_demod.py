"""
AM demodulation utilities for the communication system simulator.

This module contains:
- low-pass filtering
- standard AM envelope demodulation
- DSB-SC coherent demodulation
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def lowpass_filter(
    signal: np.ndarray,
    sample_rate: float,
    cutoff_frequency: float,
    filter_order: int = 5
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter.

    signal: input signal to filter
    sample_rate: sampling rate in Hz
    cutoff_frequency: cutoff frequency in Hz
    filter_order: order of the Butterworth filter

    Returns:
    filtered signal
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")

    if cutoff_frequency <= 0:
        raise ValueError("cutoff_frequency must be greater than 0.")

    nyquist_frequency = sample_rate / 2

    if cutoff_frequency >= nyquist_frequency:
        raise ValueError("cutoff_frequency must be less than the Nyquist frequency.")

    normalized_cutoff = cutoff_frequency / nyquist_frequency

    b, a = butter(
        N=filter_order,
        Wn=normalized_cutoff,
        btype="low"
    )

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def envelope_demodulate_am(
    received_signal: np.ndarray,
    sample_rate: float,
    cutoff_frequency: float,
    modulation_index: float = 0.8,
    filter_order: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Demodulate a standard AM signal using envelope detection.

    received_signal: noisy AM signal
    sample_rate: sampling rate in Hz
    cutoff_frequency: low-pass filter cutoff in Hz
    modulation_index: AM modulation index used during transmission
    filter_order: low-pass filter order

    Returns:
    recovered_message, envelope
    """
    if modulation_index <= 0:
        raise ValueError("modulation_index must be greater than 0.")

    analytic_signal = hilbert(received_signal)

    envelope = np.abs(analytic_signal)

    baseband_estimate = (envelope - np.mean(envelope)) / modulation_index

    recovered_message = lowpass_filter(
        signal=baseband_estimate,
        sample_rate=sample_rate,
        cutoff_frequency=cutoff_frequency,
        filter_order=filter_order
    )

    return recovered_message, envelope


def coherent_demodulate_dsb_sc(
    received_signal: np.ndarray,
    t: np.ndarray,
    carrier_frequency: float,
    sample_rate: float,
    cutoff_frequency: float,
    carrier_amplitude: float = 1.0,
    filter_order: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Demodulate a DSB-SC signal using coherent demodulation.

    received_signal: noisy DSB-SC signal
    t: time array
    carrier_frequency: carrier frequency in Hz
    sample_rate: sampling rate in Hz
    cutoff_frequency: low-pass filter cutoff in Hz
    carrier_amplitude: carrier amplitude used during transmission
    filter_order: low-pass filter order

    Returns:
    recovered_message, mixed_signal
    """
    if carrier_frequency <= 0:
        raise ValueError("carrier_frequency must be greater than 0.")

    if carrier_amplitude <= 0:
        raise ValueError("carrier_amplitude must be greater than 0.")

    local_oscillator = 2 * np.cos(2 * np.pi * carrier_frequency * t)

    mixed_signal = received_signal * local_oscillator / carrier_amplitude

    recovered_message = lowpass_filter(
        signal=mixed_signal,
        sample_rate=sample_rate,
        cutoff_frequency=cutoff_frequency,
        filter_order=filter_order
    )

    return recovered_message, mixed_signal


def test_am_demod():
    """
    Simple import test.
    """
    return "demodulation.am_demod imported correctly"