import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import upfirdn
import random

## SIGNAL PARAMETERS
# Symbol rate: 10 kbaud (10,000 symbols/second)
symbol_rate = 10e3

# Samples per symbol: 8 (good for timing recovery practice)
sps = 8

# Sample rate: 80 kHz (symbol_rate * samples_per_symbol)
sample_rate = symbol_rate * sps

# Number of symbols to generate: 1000 (good length for sync algorithms)
num_symbols = 1000

# RRC filter parameters
rrc_span = 10  # Filter span in symbols (10 symbols before and after)
rrc_beta = 0.35  # Roll-off factor (0.35 is typical for digital communications)

## IMPAIRMENT PARAMETERS (make it realistic and challenging)
# Carrier frequency offset: 500 Hz (will need to be corrected by receiver)
freq_offset = 500  # Hz

# Timing offset: random fractional symbol delay between 0 and 1 symbol
timing_offset = random.uniform(0, 1)  # Fraction of symbol period

# Phase offset: random initial phase between -π and π
phase_offset = random.uniform(-np.pi, np.pi)  # radians

# SNR in dB (signal-to-noise ratio) - lower means more noise
snr_db = 15  # dB (15 dB is moderate - good for testing algorithms)

print("=== QPSK Signal Generator ===")
print(f"Symbol rate: {symbol_rate/1e3:.1f} kbaud")
print(f"Sample rate: {sample_rate/1e3:.1f} kHz")
print(f"Samples per symbol: {sps}")
print(f"Number of symbols: {num_symbols}")
print(f"RRC roll-off (β): {rrc_beta}")
print(f"\n=== Impairments ===")
print(f"Carrier freq offset: {freq_offset} Hz")
print(f"Timing offset: {timing_offset:.3f} symbols")
print(f"Phase offset: {phase_offset:.3f} radians ({np.degrees(phase_offset):.1f}°)")
print(f"SNR: {snr_db} dB")

## STEP 1: Generate random QPSK symbols
# QPSK constellation: 4 points at ±1±j (normalized)
# Maps to phases: 45°, 135°, -135°, -45° (π/4, 3π/4, -3π/4, -π/4)
constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)

# Generate random bits and map to QPSK symbols
bits = np.random.randint(0, 4, num_symbols)  # Random integers 0-3
symbols = constellation[bits]

print(f"\nGenerated {num_symbols} random QPSK symbols")
print(f"First 10 symbols: {symbols[:10]}")

## STEP 2: Create Root Raised Cosine (RRC) pulse shaping filter
# RRC is used in real systems to limit bandwidth while allowing matched filtering at RX
def rrc_filter(sps, span, beta):
    """
    Generate Root Raised Cosine (RRC) filter coefficients

    Args:
        sps: Samples per symbol
        span: Filter span in symbols
        beta: Roll-off factor (0 < beta <= 1)

    Returns:
        h: RRC filter coefficients
    """
    num_taps = span * sps + 1
    t = np.arange(-span*sps/2, span*sps/2 + 1) / sps

    # RRC formula
    h = np.zeros(len(t))
    for i, time in enumerate(t):
        if time == 0:
            h[i] = (1 + beta*(4/np.pi - 1))
        elif abs(time) == 1/(4*beta):
            h[i] = (beta/np.sqrt(2)) * ((1 + 2/np.pi)*np.sin(np.pi/(4*beta)) +
                                         (1 - 2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            h[i] = (np.sin(np.pi*time*(1-beta)) +
                    4*beta*time*np.cos(np.pi*time*(1+beta))) / \
                   (np.pi*time*(1 - (4*beta*time)**2))

    # Normalize to unit energy
    h = h / np.sqrt(np.sum(h**2))
    return h

# Generate RRC filter
rrc = rrc_filter(sps, rrc_span, rrc_beta)
print(f"\nRRC filter: {len(rrc)} taps ({rrc_span} symbol span)")

## STEP 3: Upsample and pulse shape
# Upsample: insert (sps-1) zeros between symbols
upsampled = np.zeros(len(symbols) * sps, dtype=complex)
upsampled[::sps] = symbols

# Apply RRC filter (pulse shaping)
tx_signal = upfirdn(rrc, upsampled, up=1, down=1)

# Trim filter delay (group delay = (filter_length-1)/2 samples)
delay = (len(rrc) - 1) // 2
tx_signal = tx_signal[delay:delay + len(upsampled)]

print(f"Pulse-shaped signal: {len(tx_signal)} samples")

## STEP 4: Apply timing offset (fractional delay)
# This simulates starting to sample at a random point in the symbol
# Receiver will need to recover correct timing
from scipy.interpolate import interp1d

t_original = np.arange(len(tx_signal))
t_delayed = t_original + timing_offset * sps

# Interpolate to create fractional delay
interp_func = interp1d(t_original, tx_signal, kind='cubic',
                        bounds_error=False, fill_value=0)
tx_signal = interp_func(t_delayed[:len(t_original)])

print(f"Applied timing offset: {timing_offset:.3f} symbols")

## STEP 5: Apply carrier frequency offset (CFO)
# This simulates oscillator mismatch between TX and RX
# Creates a rotating phase: e^(j*2*π*f_offset*t)
t = np.arange(len(tx_signal)) / sample_rate
tx_signal = tx_signal * np.exp(1j * 2 * np.pi * freq_offset * t)

print(f"Applied frequency offset: {freq_offset} Hz")

## STEP 6: Apply phase offset
# Random initial phase rotation - receiver must track this
tx_signal = tx_signal * np.exp(1j * phase_offset)

print(f"Applied phase offset: {phase_offset:.3f} rad")

## STEP 7: Add AWGN (Additive White Gaussian Noise)
# Calculate noise power from desired SNR
signal_power = np.mean(np.abs(tx_signal)**2)
noise_power = signal_power / (10**(snr_db/10))
noise_std = np.sqrt(noise_power / 2)  # Divide by 2 for complex (I and Q)

# Generate complex Gaussian noise
noise_i = np.random.normal(0, noise_std, len(tx_signal))
noise_q = np.random.normal(0, noise_std, len(tx_signal))
noise = noise_i + 1j * noise_q

# Add noise to signal
rx_signal = tx_signal + noise

print(f"Added AWGN: {snr_db} dB SNR")
print(f"Signal power: {10*np.log10(signal_power):.2f} dBW")
print(f"Noise power: {10*np.log10(np.mean(np.abs(noise)**2)):.2f} dBW")

## STEP 8: Save to IQ file
output_filename = 'qpsk_signal.iq'
rx_signal.astype(np.complex64).tofile(output_filename)

print(f"\n=== Output ===")
print(f"Saved {len(rx_signal)} complex IQ samples to {output_filename}")
print(f"File size: {rx_signal.nbytes / 1e6:.2f} MB")
print(f"Duration: {len(rx_signal) / sample_rate * 1000:.1f} ms")
print(f"\nUse this file to practice:")
print("  - Timing synchronization (Mueller & Muller, Gardner, etc.)")
print("  - Carrier frequency offset (CFO) correction")
print("  - Phase recovery (Costas loop, PLL)")
print("  - Matched filtering (RRC)")
print("  - Symbol detection and demodulation")

## STEP 9: Plot constellation diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Received signal constellation (with noise and offsets)
# Decimate to approximately symbol rate for visualization
decimated = rx_signal[::sps]
ax1.scatter(decimated.real, decimated.imag, alpha=0.3, s=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('In-phase (I)')
ax1.set_ylabel('Quadrature (Q)')
ax1.set_title(f'Received Signal Constellation\n(with CFO, timing offset, phase offset, and {snr_db} dB SNR)')
ax1.axis('equal')
ax1.set_xlim([-2, 2])
ax1.set_ylim([-2, 2])

# Plot ideal QPSK constellation points
ideal = constellation * np.sqrt(2)
ax1.scatter(ideal.real, ideal.imag, c='red', s=100, marker='x',
            linewidths=3, label='Ideal symbols')
ax1.legend()

# Plot 2: Time-domain signal (real part)
time_ms = np.arange(min(1000, len(rx_signal))) / sample_rate * 1000
ax2.plot(time_ms, rx_signal[:len(time_ms)].real, linewidth=0.5)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Amplitude (I channel)')
ax2.set_title('Received Signal - Time Domain (first 1000 samples)')

plt.tight_layout()
plt.show()

print("\n=== Parameters for Receiver ===")
print(f"sample_rate = {sample_rate}")
print(f"symbol_rate = {symbol_rate}")
print(f"sps = {sps}")
print(f"rrc_beta = {rrc_beta}")
print(f"rrc_span = {rrc_span}")
