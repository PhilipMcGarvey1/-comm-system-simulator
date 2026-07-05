"""
Digital demodulation utilities for the communication system simulator.

This module supports:
- coherent BPSK passband demodulation
- decision value calculation
- recovered bit decisions
"""

import numpy as np


def bpsk_demodulate_passband(
    received_signal: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    carrier_frequency: float,
    carrier_amplitude: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coherently demodulate a passband BPSK signal.

    Receiver process:
    1. Multiply received signal by a synchronized carrier.
    2. Average over each symbol period.
    3. Decide bit 1 if the average is positive.
    4. Decide bit 0 if the average is negative.

    Returns:
    recovered_bits, decision_values
    """
    if received_signal.size == 0:
        raise ValueError("received_signal must not be empty.")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")

    if symbol_rate <= 0:
        raise ValueError("symbol_rate must be greater than 0.")

    if carrier_frequency <= 0:
        raise ValueError("carrier_frequency must be greater than 0.")

    if carrier_amplitude <= 0:
        raise ValueError("carrier_amplitude must be greater than 0.")

    samples_per_symbol = int(sample_rate / symbol_rate)

    if samples_per_symbol <= 0:
        raise ValueError("sample_rate must be greater than symbol_rate.")

    number_of_complete_symbols = received_signal.size // samples_per_symbol

    usable_samples = number_of_complete_symbols * samples_per_symbol

    received_signal = received_signal[:usable_samples]

    t = np.arange(usable_samples) / sample_rate

    local_carrier = 2 * np.cos(2 * np.pi * carrier_frequency * t) / carrier_amplitude

    mixed_signal = received_signal * local_carrier

    symbol_matrix = mixed_signal.reshape(
        number_of_complete_symbols,
        samples_per_symbol
    )

    decision_values = np.mean(symbol_matrix, axis=1)

    recovered_bits = (decision_values >= 0).astype(np.int8)

    return recovered_bits, decision_values


def test_digital_demod():
    """
    Simple import test.
    """
    return "demodulation.digital_demod imported correctly"