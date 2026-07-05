"""
BER analysis utilities for the communication system simulator.

BER means Bit Error Rate.

BER = number of bit errors / total number of transmitted bits
"""

import numpy as np


def count_bit_errors(
    transmitted_bits: np.ndarray,
    received_bits: np.ndarray
) -> int:
    """
    Count how many transmitted bits were recovered incorrectly.
    """
    if transmitted_bits.shape != received_bits.shape:
        raise ValueError("transmitted_bits and received_bits must have the same shape.")

    if transmitted_bits.size == 0:
        raise ValueError("bit arrays must not be empty.")

    errors = transmitted_bits != received_bits

    return int(np.sum(errors))


def calculate_ber(
    transmitted_bits: np.ndarray,
    received_bits: np.ndarray
) -> float:
    """
    Calculate bit error rate.

    BER = bit errors / total bits
    """
    bit_errors = count_bit_errors(
        transmitted_bits=transmitted_bits,
        received_bits=received_bits
    )

    total_bits = transmitted_bits.size

    return float(bit_errors / total_bits)


def test_ber():
    """
    Simple import test.
    """
    return "analysis.ber imported correctly"