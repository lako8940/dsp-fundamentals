import numpy as np
from scipy.signal import resample_poly, firwin
import matplotlib.pyplot as plt

# Read and process signal
x = np.fromfile('./fm_capture.iq', dtype=np.complex64)
print(f"Loaded {len(x)} IQ samples")

# Normalize
x = x / np.abs(x).max()

# Quadrature Demod
x = 0.5 * np.angle(x[0:-1] * np.conj(x[1:]))
print(f"After FM demod: {len(x)} samples, range: [{x.min():.3f}, {x.max():.3f}]")

# Freq shift to baseband (RDS at 57 kHz)
sample_rate = 250e3
N = len(x)
f_o = -57e3
t = np.arange(N)/sample_rate
x = x * np.exp(2j*np.pi*f_o*t)

# Low-Pass Filter
taps = firwin(numtaps=101, cutoff=7.5e3, fs=sample_rate)
x = np.convolve(x, taps, 'valid')

# Decimate by 10
x = x[::10]
sample_rate = 25e3

# Resample to 19kHz
x = resample_poly(x, 19, 25)
sample_rate = 19e3

print(f"\nSignal before timing recovery:")
print(f"  Length: {len(x)} samples")
print(f"  Sample rate: {sample_rate} Hz")
print(f"  Expected symbol rate: 1187.5 baud")
print(f"  Samples per symbol: {sample_rate / 1187.5:.2f}")
print(f"  Signal power: {np.mean(np.abs(x)**2):.6f}")
print(f"  Signal amplitude range: [{np.abs(x).min():.6f}, {np.abs(x).max():.6f}]")
print(f"  Signal mean (DC offset): {np.abs(np.mean(x)):.6f}")

# Check for strong periodic components
fft_data = np.fft.fft(x[:8192])
freqs = np.fft.fftfreq(8192, 1/sample_rate)
peak_idx = np.argmax(np.abs(fft_data[:4096]))
peak_freq = freqs[peak_idx]
print(f"\nSpectral analysis:")
print(f"  Dominant frequency component: {peak_freq:.1f} Hz")
print(f"  Expected RDS carrier (after downconv): ~0 Hz")

# Create detailed plots
fig = plt.figure(figsize=(16, 10))

# 1. Time domain - Real part
ax1 = plt.subplot(3, 3, 1)
ax1.plot(np.real(x[:1000]))
ax1.set_title('Signal - Real Part (first 1000 samples)')
ax1.set_xlabel('Sample')
ax1.grid(True)

# 2. Time domain - Imag part
ax2 = plt.subplot(3, 3, 2)
ax2.plot(np.imag(x[:1000]))
ax2.set_title('Signal - Imag Part (first 1000 samples)')
ax2.set_xlabel('Sample')
ax2.grid(True)

# 3. Magnitude
ax3 = plt.subplot(3, 3, 3)
ax3.plot(np.abs(x[:1000]))
ax3.set_title('Signal Magnitude (first 1000 samples)')
ax3.set_xlabel('Sample')
ax3.grid(True)

# 4. Spectrum
ax4 = plt.subplot(3, 3, 4)
ax4.plot(freqs[:4096], 20*np.log10(np.abs(fft_data[:4096])))
ax4.set_title('Spectrum')
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Magnitude (dB)')
ax4.grid(True)
ax4.axvline(1187.5, color='r', linestyle='--', label='Symbol rate')
ax4.legend()

# 5. Constellation (without timing recovery)
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(np.real(x[::16][:1000]), np.imag(x[::16][:1000]), alpha=0.3, s=1)
ax5.set_title('Constellation (decimated by 16)')
ax5.set_xlabel('I')
ax5.set_ylabel('Q')
ax5.grid(True)
ax5.axis('equal')

# 6. Autocorrelation to find symbol period
ax6 = plt.subplot(3, 3, 6)
autocorr = np.correlate(x[:2000], x[:2000], mode='same')
ax6.plot(np.abs(autocorr[1000:1500]))
ax6.set_title('Autocorrelation')
ax6.set_xlabel('Lag (samples)')
ax6.axvline(sample_rate/1187.5, color='r', linestyle='--', label=f'Expected: {sample_rate/1187.5:.1f}')
ax6.legend()
ax6.grid(True)

# 7. Phase
ax7 = plt.subplot(3, 3, 7)
ax7.plot(np.angle(x[:1000]))
ax7.set_title('Phase (first 1000 samples)')
ax7.set_xlabel('Sample')
ax7.set_ylabel('Phase (radians)')
ax7.grid(True)

# 8. Histogram of magnitudes
ax8 = plt.subplot(3, 3, 8)
ax8.hist(np.abs(x), bins=100, alpha=0.7)
ax8.set_title('Magnitude Distribution')
ax8.set_xlabel('Magnitude')
ax8.set_ylabel('Count')
ax8.grid(True)

# 9. Statistics
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

# Calculate SNR estimate
signal_power = np.mean(np.abs(x)**2)
noise_floor = np.percentile(np.abs(x)**2, 10)  # Rough estimate
snr_db = 10 * np.log10(signal_power / noise_floor) if noise_floor > 0 else 0

stats_text = f"""
Signal Statistics:
━━━━━━━━━━━━━━━━━━━━━━
Samples: {len(x)}
Sample rate: {sample_rate:.0f} Hz
Symbol rate: 1187.5 baud
Samples/symbol: {sample_rate/1187.5:.2f}

Amplitude:
  Mean: {np.abs(x).mean():.6f}
  Std:  {np.abs(x).std():.6f}
  Max:  {np.abs(x).max():.6f}

Signal quality:
  Power: {signal_power:.6f}
  Est. SNR: {snr_db:.1f} dB
  DC offset: {np.abs(np.mean(x)):.6f}

Phase characteristics:
  Mean: {np.mean(np.angle(x)):.3f} rad
  Std:  {np.std(np.angle(x)):.3f} rad

Dominant freq: {peak_freq:.1f} Hz
Expected: ~0 Hz (baseband)
"""
ax9.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig('rds_signal_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved analysis to rds_signal_analysis.png")

# Recommendations
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if np.abs(np.mean(x)) > 0.01:
    print("⚠ DC offset detected! Consider removing it before timing recovery")

if np.abs(x).std() < 0.01:
    print("⚠ Very low signal amplitude variation")

if peak_freq > 100:
    print(f"⚠ Strong frequency component at {peak_freq:.1f} Hz")
    print("  This suggests incomplete frequency offset correction")
    print(f"  Try adjusting f_o from -57 kHz to {-57e3 - peak_freq:.0f} Hz")

if sample_rate / 1187.5 < 15 or sample_rate / 1187.5 > 17:
    print(f"⚠ Unusual samples per symbol: {sample_rate/1187.5:.2f}")
    print("  Expected: ~16 samples/symbol")

print("\nKey insight: For M&M to work, the signal must:")
print("1. Be at baseband (near 0 Hz) - check spectrum")
print("2. Have good SNR - check constellation")
print("3. Have ~16 samples/symbol - check")
print("4. Have minimal DC offset - check")
