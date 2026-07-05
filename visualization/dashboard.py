"""
Streamlit dashboard for the End-to-End Communication System Simulator.

Phase 9 purpose:
- Create an interactive dashboard
- Visualize analog AM / DSB-SC communication
- Visualize digital BPSK communication
- Allow parameter changes from the sidebar
- Show time-domain, frequency-domain, and BER results
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Make imports work when running:
# streamlit run visualization/dashboard.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from signals.generator import (
    SignalConfig,
    create_time_axis,
    multi_tone_signal,
    normalize_signal
)

from modulation.am import (
    standard_am_modulate,
    dsb_sc_modulate
)

from signals.noise import add_awgn

from demodulation.am_demod import (
    envelope_demodulate_am,
    coherent_demodulate_dsb_sc
)

from analysis.snr import (
    calculate_snr_db,
    mean_squared_error,
    root_mean_squared_error,
    normalized_mean_squared_error
)

from analysis.fft import (
    compute_single_sided_fft,
    amplitude_to_db
)

from modulation.digital import (
    generate_random_bits,
    bits_to_bpsk_symbols,
    bpsk_modulate_passband
)

from demodulation.digital_demod import bpsk_demodulate_passband

from analysis.ber import (
    count_bit_errors,
    calculate_ber
)


def add_awgn_for_bpsk_ebn0(
    signal: np.ndarray,
    samples_per_symbol: int,
    ebn0_db: float,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add AWGN to a passband BPSK signal based on Eb/N0.

    This is used for digital BER simulations.
    """
    ebn0_linear = 10 ** (ebn0_db / 10)

    noise_variance = samples_per_symbol / (4 * ebn0_linear)

    noise_std = np.sqrt(noise_variance)

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=signal.shape
    )

    noisy_signal = signal + noise

    return noisy_signal, noise


def run_analog_simulation(
    sample_rate: float,
    duration: float,
    carrier_frequency: float,
    modulation_index: float,
    target_snr_db: float,
    lowpass_cutoff: float
) -> dict:
    """
    Run the analog AM / DSB-SC communication simulation.
    """
    config = SignalConfig(
        sample_rate=sample_rate,
        duration=duration
    )

    t = create_time_axis(config)

    message = multi_tone_signal(
        t=t,
        tones=[
            (500, 1.0),
            (1000, 0.5),
            (2000, 0.25)
        ]
    )

    message = normalize_signal(
        signal=message,
        target_peak=1.0
    )

    am_signal = standard_am_modulate(
        message=message,
        t=t,
        carrier_frequency=carrier_frequency,
        modulation_index=modulation_index,
        carrier_amplitude=1.0
    )

    dsb_sc_signal = dsb_sc_modulate(
        message=message,
        t=t,
        carrier_frequency=carrier_frequency,
        carrier_amplitude=1.0
    )

    noisy_am_signal, am_noise = add_awgn(
        signal=am_signal,
        snr_db=target_snr_db,
        seed=42
    )

    noisy_dsb_sc_signal, dsb_sc_noise = add_awgn(
        signal=dsb_sc_signal,
        snr_db=target_snr_db,
        seed=42
    )

    recovered_am_message, am_envelope = envelope_demodulate_am(
        received_signal=noisy_am_signal,
        sample_rate=sample_rate,
        cutoff_frequency=lowpass_cutoff,
        modulation_index=modulation_index,
        filter_order=5
    )

    recovered_dsb_sc_message, mixed_dsb_sc_signal = coherent_demodulate_dsb_sc(
        received_signal=noisy_dsb_sc_signal,
        t=t,
        carrier_frequency=carrier_frequency,
        sample_rate=sample_rate,
        cutoff_frequency=lowpass_cutoff,
        carrier_amplitude=1.0,
        filter_order=5
    )

    recovered_am_message = normalize_signal(
        signal=recovered_am_message,
        target_peak=1.0
    )

    recovered_dsb_sc_message = normalize_signal(
        signal=recovered_dsb_sc_message,
        target_peak=1.0
    )

    measured_am_snr_db = calculate_snr_db(
        clean_signal=am_signal,
        noisy_signal=noisy_am_signal
    )

    measured_dsb_sc_snr_db = calculate_snr_db(
        clean_signal=dsb_sc_signal,
        noisy_signal=noisy_dsb_sc_signal
    )

    am_mse = mean_squared_error(
        reference_signal=message,
        test_signal=recovered_am_message
    )

    dsb_sc_mse = mean_squared_error(
        reference_signal=message,
        test_signal=recovered_dsb_sc_message
    )

    am_rmse = root_mean_squared_error(
        reference_signal=message,
        test_signal=recovered_am_message
    )

    dsb_sc_rmse = root_mean_squared_error(
        reference_signal=message,
        test_signal=recovered_dsb_sc_message
    )

    am_nmse = normalized_mean_squared_error(
        reference_signal=message,
        test_signal=recovered_am_message
    )

    dsb_sc_nmse = normalized_mean_squared_error(
        reference_signal=message,
        test_signal=recovered_dsb_sc_message
    )

    return {
        "t": t,
        "message": message,
        "am_signal": am_signal,
        "dsb_sc_signal": dsb_sc_signal,
        "noisy_am_signal": noisy_am_signal,
        "noisy_dsb_sc_signal": noisy_dsb_sc_signal,
        "recovered_am_message": recovered_am_message,
        "recovered_dsb_sc_message": recovered_dsb_sc_message,
        "measured_am_snr_db": measured_am_snr_db,
        "measured_dsb_sc_snr_db": measured_dsb_sc_snr_db,
        "am_mse": am_mse,
        "dsb_sc_mse": dsb_sc_mse,
        "am_rmse": am_rmse,
        "dsb_sc_rmse": dsb_sc_rmse,
        "am_nmse": am_nmse,
        "dsb_sc_nmse": dsb_sc_nmse
    }


def plot_analog_time_domain(results: dict, time_limit: float):
    """
    Create analog time-domain plot.
    """
    t = results["t"]
    plot_indices = t <= time_limit

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(t[plot_indices], results["message"][plot_indices])
    axes[0].set_title("Original Baseband Message")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(t[plot_indices], results["noisy_am_signal"][plot_indices])
    axes[1].set_title("Noisy Standard AM Signal")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].plot(t[plot_indices], results["message"][plot_indices], label="Original")
    axes[2].plot(t[plot_indices], results["recovered_am_message"][plot_indices], label="Recovered AM", alpha=0.8)
    axes[2].set_title("AM Recovery")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(t[plot_indices], results["message"][plot_indices], label="Original")
    axes[3].plot(t[plot_indices], results["recovered_dsb_sc_message"][plot_indices], label="Recovered DSB-SC", alpha=0.8)
    axes[3].set_title("DSB-SC Recovery")
    axes[3].set_xlabel("Time [seconds]")
    axes[3].set_ylabel("Amplitude")
    axes[3].legend()
    axes[3].grid(True)

    fig.tight_layout()

    return fig


def plot_analog_frequency_domain(results: dict, sample_rate: float):
    """
    Create analog frequency-domain FFT plot.
    """
    message_freqs, message_mags = compute_single_sided_fft(
        signal=results["message"],
        sample_rate=sample_rate
    )

    am_freqs, am_mags = compute_single_sided_fft(
        signal=results["am_signal"],
        sample_rate=sample_rate
    )

    dsb_sc_freqs, dsb_sc_mags = compute_single_sided_fft(
        signal=results["dsb_sc_signal"],
        sample_rate=sample_rate
    )

    recovered_freqs, recovered_mags = compute_single_sided_fft(
        signal=results["recovered_dsb_sc_message"],
        sample_rate=sample_rate
    )

    message_db = amplitude_to_db(message_mags)
    am_db = amplitude_to_db(am_mags)
    dsb_sc_db = amplitude_to_db(dsb_sc_mags)
    recovered_db = amplitude_to_db(recovered_mags)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8))

    axes[0].plot(message_freqs, message_db)
    axes[0].set_title("Baseband Message Spectrum")
    axes[0].set_xlim(0, 5000)
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].grid(True)

    axes[1].plot(am_freqs, am_db)
    axes[1].set_title("Standard AM Spectrum")
    axes[1].set_xlim(0, 15000)
    axes[1].set_ylabel("Magnitude [dB]")
    axes[1].grid(True)

    axes[2].plot(dsb_sc_freqs, dsb_sc_db)
    axes[2].set_title("DSB-SC Spectrum")
    axes[2].set_xlim(0, 15000)
    axes[2].set_ylabel("Magnitude [dB]")
    axes[2].grid(True)

    axes[3].plot(recovered_freqs, recovered_db)
    axes[3].set_title("Recovered DSB-SC Message Spectrum")
    axes[3].set_xlim(0, 5000)
    axes[3].set_xlabel("Frequency [Hz]")
    axes[3].set_ylabel("Magnitude [dB]")
    axes[3].grid(True)

    fig.tight_layout()

    return fig


def run_bpsk_ber_simulation(
    number_of_bits: int,
    sample_rate: float,
    symbol_rate: float,
    carrier_frequency: float,
    ebn0_values_db: list[int]
) -> tuple[pd.DataFrame, dict]:
    """
    Run BPSK BER simulation over multiple Eb/N0 values.
    """
    transmitted_bits = generate_random_bits(
        number_of_bits=number_of_bits,
        seed=42
    )

    transmitted_symbols = bits_to_bpsk_symbols(transmitted_bits)

    t, baseband_waveform, clean_bpsk_signal = bpsk_modulate_passband(
        symbols=transmitted_symbols,
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        carrier_frequency=carrier_frequency,
        carrier_amplitude=1.0
    )

    samples_per_symbol = int(sample_rate / symbol_rate)

    ber_results = []

    example_ebn0_db = ebn0_values_db[len(ebn0_values_db) // 2]
    example_noisy_signal = None
    example_decision_values = None
    example_recovered_bits = None

    for ebn0_db in ebn0_values_db:
        noisy_bpsk_signal, noise = add_awgn_for_bpsk_ebn0(
            signal=clean_bpsk_signal,
            samples_per_symbol=samples_per_symbol,
            ebn0_db=ebn0_db,
            seed=100 + ebn0_db
        )

        recovered_bits, decision_values = bpsk_demodulate_passband(
            received_signal=noisy_bpsk_signal,
            sample_rate=sample_rate,
            symbol_rate=symbol_rate,
            carrier_frequency=carrier_frequency,
            carrier_amplitude=1.0
        )

        transmitted_bits_used = transmitted_bits[:recovered_bits.size]

        bit_errors = count_bit_errors(
            transmitted_bits=transmitted_bits_used,
            received_bits=recovered_bits
        )

        ber = calculate_ber(
            transmitted_bits=transmitted_bits_used,
            received_bits=recovered_bits
        )

        ber_results.append(
            {
                "ebn0_db": ebn0_db,
                "number_of_bits": transmitted_bits_used.size,
                "bit_errors": bit_errors,
                "ber": ber
            }
        )

        if ebn0_db == example_ebn0_db:
            example_noisy_signal = noisy_bpsk_signal
            example_decision_values = decision_values
            example_recovered_bits = recovered_bits

    ber_table = pd.DataFrame(ber_results)

    example = {
        "t": t,
        "clean_bpsk_signal": clean_bpsk_signal,
        "noisy_bpsk_signal": example_noisy_signal,
        "decision_values": example_decision_values,
        "transmitted_bits": transmitted_bits,
        "recovered_bits": example_recovered_bits,
        "samples_per_symbol": samples_per_symbol,
        "example_ebn0_db": example_ebn0_db
    }

    return ber_table, example


def plot_bpsk_results(ber_table: pd.DataFrame, example: dict):
    """
    Plot BPSK time-domain example and BER curve.
    """
    bits_to_plot = 8
    samples_to_plot = bits_to_plot * example["samples_per_symbol"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9))

    axes[0].plot(
        example["t"][:samples_to_plot],
        example["clean_bpsk_signal"][:samples_to_plot]
    )
    axes[0].set_title("Clean Passband BPSK Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(
        example["t"][:samples_to_plot],
        example["noisy_bpsk_signal"][:samples_to_plot]
    )
    axes[1].set_title(f"Noisy Passband BPSK Signal, Eb/N0 = {example['example_ebn0_db']} dB")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].stem(
        range(bits_to_plot),
        example["decision_values"][:bits_to_plot]
    )
    axes[2].set_title("BPSK Receiver Decision Values")
    axes[2].set_ylabel("Decision")
    axes[2].grid(True)

    axes[3].semilogy(
        ber_table["ebn0_db"],
        ber_table["ber"],
        marker="o"
    )
    axes[3].set_title("BPSK BER vs Eb/N0")
    axes[3].set_xlabel("Eb/N0 [dB]")
    axes[3].set_ylabel("Bit Error Rate")
    axes[3].grid(True, which="both")

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Communication System Simulator",
    layout="wide"
)

st.title("End-to-End Communication System Simulator")

st.markdown(
    """
    This dashboard demonstrates analog and digital communication system concepts:

    - Baseband signal generation
    - AM and DSB-SC modulation
    - AWGN noise channel
    - Demodulation and filtering
    - SNR and error metrics
    - FFT frequency-domain analysis
    - BPSK modulation and BER analysis
    """
)

with st.sidebar:
    st.header("Simulation Controls")

    mode = st.selectbox(
        "Choose simulation mode",
        [
            "Analog AM / DSB-SC",
            "Digital BPSK"
        ]
    )

    st.divider()

    if mode == "Analog AM / DSB-SC":
        st.subheader("Analog Settings")

        sample_rate = st.slider(
            "Sample rate [Hz]",
            min_value=50_000,
            max_value=200_000,
            value=100_000,
            step=10_000
        )

        duration = st.slider(
            "Duration [seconds]",
            min_value=0.01,
            max_value=0.05,
            value=0.02,
            step=0.01
        )

        carrier_frequency = st.slider(
            "Carrier frequency [Hz]",
            min_value=5_000,
            max_value=20_000,
            value=10_000,
            step=1_000
        )

        modulation_index = st.slider(
            "AM modulation index",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1
        )

        target_snr_db = st.slider(
            "Channel SNR [dB]",
            min_value=0,
            max_value=40,
            value=20,
            step=5
        )

        lowpass_cutoff = st.slider(
            "Receiver low-pass cutoff [Hz]",
            min_value=1_000,
            max_value=5_000,
            value=3_000,
            step=500
        )

        time_limit = st.slider(
            "Time plot limit [seconds]",
            min_value=0.003,
            max_value=0.02,
            value=0.01,
            step=0.001
        )

    else:
        st.subheader("Digital BPSK Settings")

        number_of_bits = st.slider(
            "Number of bits",
            min_value=1_000,
            max_value=50_000,
            value=10_000,
            step=1_000
        )

        sample_rate = st.slider(
            "Sample rate [Hz]",
            min_value=50_000,
            max_value=200_000,
            value=100_000,
            step=10_000
        )

        symbol_rate = st.slider(
            "Symbol rate [symbols/sec]",
            min_value=500,
            max_value=5_000,
            value=1_000,
            step=500
        )

        carrier_frequency = st.slider(
            "Carrier frequency [Hz]",
            min_value=5_000,
            max_value=20_000,
            value=10_000,
            step=1_000
        )

        max_ebn0_db = st.slider(
            "Maximum Eb/N0 [dB]",
            min_value=4,
            max_value=12,
            value=10,
            step=1
        )


if mode == "Analog AM / DSB-SC":
    st.header("Analog AM / DSB-SC Simulation")

    results = run_analog_simulation(
        sample_rate=sample_rate,
        duration=duration,
        carrier_frequency=carrier_frequency,
        modulation_index=modulation_index,
        target_snr_db=target_snr_db,
        lowpass_cutoff=lowpass_cutoff
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Measured AM SNR", f"{results['measured_am_snr_db']:.2f} dB")
    col2.metric("Measured DSB-SC SNR", f"{results['measured_dsb_sc_snr_db']:.2f} dB")
    col3.metric("AM MSE", f"{results['am_mse']:.6f}")
    col4.metric("DSB-SC MSE", f"{results['dsb_sc_mse']:.6f}")

    metric_table = pd.DataFrame(
        [
            {
                "System": "AM",
                "MSE": results["am_mse"],
                "RMSE": results["am_rmse"],
                "NMSE": results["am_nmse"],
                "Measured SNR [dB]": results["measured_am_snr_db"]
            },
            {
                "System": "DSB-SC",
                "MSE": results["dsb_sc_mse"],
                "RMSE": results["dsb_sc_rmse"],
                "NMSE": results["dsb_sc_nmse"],
                "Measured SNR [dB]": results["measured_dsb_sc_snr_db"]
            }
        ]
    )

    st.subheader("Performance Metrics")
    st.dataframe(metric_table, use_container_width=True)

    st.subheader("Time-Domain Signals")
    time_fig = plot_analog_time_domain(
        results=results,
        time_limit=time_limit
    )
    st.pyplot(time_fig)

    st.subheader("Frequency-Domain Spectra")
    fft_fig = plot_analog_frequency_domain(
        results=results,
        sample_rate=sample_rate
    )
    st.pyplot(fft_fig)

else:
    st.header("Digital BPSK Simulation")

    ebn0_values_db = list(range(0, max_ebn0_db + 1))

    ber_table, example = run_bpsk_ber_simulation(
        number_of_bits=number_of_bits,
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        carrier_frequency=carrier_frequency,
        ebn0_values_db=ebn0_values_db
    )

    final_ber = ber_table.iloc[-1]["ber"]
    first_ber = ber_table.iloc[0]["ber"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Bits Simulated", f"{number_of_bits}")
    col2.metric("Symbol Rate", f"{symbol_rate} sym/s")
    col3.metric("BER at 0 dB", f"{first_ber:.6f}")
    col4.metric(f"BER at {max_ebn0_db} dB", f"{final_ber:.6f}")

    st.subheader("BER Results")
    st.dataframe(ber_table, use_container_width=True)

    st.subheader("BPSK Waveforms and BER Curve")
    bpsk_fig = plot_bpsk_results(
        ber_table=ber_table,
        example=example
    )
    st.pyplot(bpsk_fig)

    st.subheader("Bit Recovery Example")

    comparison_bits = 30

    bit_comparison = pd.DataFrame(
        {
            "Bit Index": list(range(comparison_bits)),
            "Transmitted Bit": example["transmitted_bits"][:comparison_bits],
            "Recovered Bit": example["recovered_bits"][:comparison_bits],
            "Decision Value": example["decision_values"][:comparison_bits]
        }
    )

    st.dataframe(bit_comparison, use_container_width=True)