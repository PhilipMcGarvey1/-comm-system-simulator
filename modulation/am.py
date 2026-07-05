"""
AM modulation utilities for the communication system simulator.

This module contains:
- carrier generation
- standard AM modulation
- DSB-SC modulation
"""

import numpy as np


def generate_carrier(
    t: np.ndarray,
    carrier_frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0
) -> np.ndarray:
    """
    Generate a cosine carrier wave.

    t: time array
    carrier_frequency: carrier frequency in Hz
    amplitude: carrier amplitude
    phase: phase shift in radians
    """
    if carrier_frequency <= 0:
        raise ValueError("carrier_frequency must be greater than 0.")

    return amplitude * np.cos(2 * np.pi * carrier_frequency * t + phase)


def standard_am_modulate(
    message: np.ndarray,
    t: np.ndarray,
    carrier_frequency: float,
    modulation_index: float = 0.8,
    carrier_amplitude: float = 1.0
) -> np.ndarray:
    """
    Perform standard AM modulation.

    Formula:
    s_AM(t) = Ac * [1 + mu * m(t)] * cos(2*pi*fc*t)

    message: normalized message signal, usually between -1 and 1
    t: time array
    carrier_frequency: carrier frequency in Hz
    modulation_index: controls modulation depth
    carrier_amplitude: carrier amplitude
    """
    if modulation_index < 0:
        raise ValueError("modulation_index must be nonnegative.")

    carrier = generate_carrier(
        t=t,
        carrier_frequency=carrier_frequency,
        amplitude=carrier_amplitude
    )

    return (1 + modulation_index * message) * carrier


def dsb_sc_modulate(
    message: np.ndarray,
    t: np.ndarray,
    carrier_frequency: float,
    carrier_amplitude: float = 1.0
) -> np.ndarray:
    """
    Perform DSB-SC modulation.

    Formula:
    s_DSBSC(t) = Ac * m(t) * cos(2*pi*fc*t)

    message: baseband message signal
    t: time array
    carrier_frequency: carrier frequency in Hz
    carrier_amplitude: carrier amplitude
    """
    carrier = generate_carrier(
        t=t,
        carrier_frequency=carrier_frequency,
        amplitude=carrier_amplitude
    )

    return message * carrier


def test_am():
    """
    Simple import test.
    """
    return "modulation.am imported correctly"