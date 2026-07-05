"""
Digital modulation utilities for the communication system simulator.

This module currently supports:
- random bit generation
- BPSK symbol mapping
- rectangular pulse shaping
- passband BPSK modulation
"""

import numpy as np


def generate_random_bits(
    number_of_bits: int,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate a random binary bit sequence.

    number_of_bits: number of bits to generate
    seed: optional random seed for repeatable results

    Returns:
    NumPy array containing 0s and 1s
    """
    if number_of_bits <= 0:
        raise ValueError("number_of_bits must be greater than 0.")

    rng = np.random.default_rng(seed)

    bits = rng.integers(
        low=0,
        high=2,
        size=number_of_bits,
        dtype=np.int8
    )

    return bits


def bits_to_bpsk_symbols(bits: np.ndarray) -> np.ndarray:
    """
    Convert bits to BPSK symbols.

    Mapping:
    bit 0 -> -1
    bit 1 -> +1

    bits: array of 0s and 1s

    Returns:
    array of BPSK symbols
    """
    if bits.size == 0:
        raise ValueError("bits must not be empty.")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must only contain 0s and 1s.")

    symbols = 2 * bits - 1

    return symbols.astype(float)


def create_rectangular_pulse_train(
    symbols: np.ndarray,
    samples_per_symbol: int
) -> np.ndarray:
    """
    Create a rectangular pulse train from digital symbols.

    Each symbol is repeated samples_per_symbol times.

    Example:
    symbols = [+1, -1, +1]
    samples_per_symbol = 4

    output = [+1, +1, +1, +1, -1, -1, -1, -1, +1, +1, +1, +1]
    """
    if symbols.size == 0:
        raise ValueError("symbols must not be empty.")

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be greater than 0.")

    return np.repeat(symbols, samples_per_symbol)


def bpsk_modulate_passband(
    symbols: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    carrier_frequency: float,
    carrier_amplitude: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a passband BPSK signal.

    symbols: BPSK symbols, usually -1 and +1
    sample_rate: samples per second in Hz
    symbol_rate: symbols per second in Hz
    carrier_frequency: carrier frequency in Hz
    carrier_amplitude: carrier amplitude

    Returns:
    t, baseband_waveform, bpsk_signal
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")

    if symbol_rate <= 0:
        raise ValueError("symbol_rate must be greater than 0.")

    if carrier_frequency <= 0:
        raise ValueError("carrier_frequency must be greater than 0.")

    samples_per_symbol = int(sample_rate / symbol_rate)

    if samples_per_symbol <= 0:
        raise ValueError("sample_rate must be greater than symbol_rate.")

    actual_symbol_rate = sample_rate / samples_per_symbol

    if abs(actual_symbol_rate - symbol_rate) > 1e-9:
        print(
            f"Warning: symbol_rate adjusted from {symbol_rate} Hz "
            f"to {actual_symbol_rate} Hz because samples_per_symbol must be an integer."
        )

    baseband_waveform = create_rectangular_pulse_train(
        symbols=symbols,
        samples_per_symbol=samples_per_symbol
    )

    number_of_samples = baseband_waveform.size

    t = np.arange(number_of_samples) / sample_rate

    carrier = carrier_amplitude * np.cos(
        2 * np.pi * carrier_frequency * t
    )

    bpsk_signal = baseband_waveform * carrier

    return t, baseband_waveform, bpsk_signal


def test_digital():
    """
    Simple import test.
    """
    return "modulation.digital imported correctly"