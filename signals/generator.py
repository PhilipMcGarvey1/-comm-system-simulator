"""
Signal generation utilities for the communication system simulator.

This module creates time axes, sine waves, and multi-tone baseband signals.
These signals will later be used as inputs to modulation systems.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SignalConfig:
    """
    Stores basic signal generation settings.

    sample_rate: number of samples per second, measured in Hz
    duration: total length of the signal, measured in seconds
    """
    sample_rate: float = 100_000
    duration: float = 0.02


def create_time_axis(config: SignalConfig) -> np.ndarray:
    """
    Create a time array from 0 to the signal duration.

    Example:
    sample_rate = 100,000 Hz
    duration = 0.02 seconds

    total samples = 100,000 * 0.02 = 2,000 samples
    """
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0.")

    if config.duration <= 0:
        raise ValueError("duration must be greater than 0.")

    dt = 1 / config.sample_rate
    return np.arange(0, config.duration, dt)


def sine_wave(
    t: np.ndarray,
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0
) -> np.ndarray:
    """
    Generate a sine wave.

    t: time array
    frequency: sine wave frequency in Hz
    amplitude: peak amplitude of the sine wave
    phase: phase shift in radians
    """
    if frequency < 0:
        raise ValueError("frequency must be nonnegative.")

    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def multi_tone_signal(
    t: np.ndarray,
    tones: list[tuple[float, float]]
) -> np.ndarray:
    """
    Generate a signal made from multiple sine waves.

    tones should be a list of:
    (frequency, amplitude)

    Example:
    tones = [
        (500, 1.0),
        (1000, 0.5),
        (2000, 0.25)
    ]
    """
    signal = np.zeros_like(t)

    for frequency, amplitude in tones:
        signal += sine_wave(
            t=t,
            frequency=frequency,
            amplitude=amplitude
        )

    return signal


def normalize_signal(signal: np.ndarray, target_peak: float = 1.0) -> np.ndarray:
    """
    Normalize a signal so its largest absolute value equals target_peak.

    This keeps the signal amplitude controlled before modulation.
    """
    if target_peak <= 0:
        raise ValueError("target_peak must be greater than 0.")

    peak = np.max(np.abs(signal))

    if peak == 0:
        return signal

    return target_peak * signal / peak


def test_generator():
    """
    Simple Phase 0 compatibility test.
    """
    return "signals.generator imported correctly"