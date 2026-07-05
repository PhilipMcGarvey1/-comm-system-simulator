"""
End-to-End Communication System Simulator

Final project launcher / system check.

This script verifies that the major project modules can be imported.
The main interactive demo is launched with:

    streamlit run visualization/dashboard.py
"""

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
    amplitude_to_db,
    find_dominant_frequencies
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


def main():
    print("End-to-End Communication System Simulator")
    print("-----------------------------------------")
    print("Project modules imported successfully.")
    print()
    print("Implemented features:")
    print("- Signal generation")
    print("- AM modulation")
    print("- DSB-SC modulation")
    print("- AWGN noise modeling")
    print("- AM envelope demodulation")
    print("- DSB-SC coherent demodulation")
    print("- Low-pass filtering")
    print("- SNR and error metrics")
    print("- FFT frequency-domain analysis")
    print("- BPSK modulation")
    print("- BPSK demodulation")
    print("- BER analysis")
    print("- Streamlit dashboard")
    print()
    print("To launch the dashboard, run:")
    print("streamlit run visualization/dashboard.py")
    print()
    print("If streamlit is not recognized, run:")
    print("python -m streamlit run visualization/dashboard.py")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()