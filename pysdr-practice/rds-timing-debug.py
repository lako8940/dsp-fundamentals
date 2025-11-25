import numpy as np
from scipy.signal import resample_poly, firwin
import matplotlib.pyplot as plt

# Read and process signal (same as fm-rds-demod.py)
x = np.fromfile('./fm_capture.iq', dtype=np.complex64)
print(f"Loaded {len(x)} IQ samples")

# Normalize
x = x / np.abs(x).max()

# Quadrature Demod
x = 0.5 * np.angle(x[0:-1] * np.conj(x[1:]))
print(f"After FM demod: {len(x)} samples")

# Freq shift to baseband (RDS at 57 kHz)
sample_rate = 250e3
N = len(x)
f_o = -57e3
t = np.arange(N)/sample_rate
x = x * np.exp(2j*np.pi*f_o*t)
print(f"After freq shift: {len(x)} samples")

# Low-Pass Filter
taps = firwin(numtaps=101, cutoff=7.5e3, fs=sample_rate)
x = np.convolve(x, taps, 'valid')
print(f"After LPF: {len(x)} samples")

# Decimate by 10
x = x[::10]
sample_rate = 25e3
print(f"After decimation: {len(x)} samples @ 25 kHz")

# Resample to 19kHz
x = resample_poly(x, 19, 25)
sample_rate = 19e3
print(f"After resampling: {len(x)} samples @ 19 kHz")

print("\n" + "="*60)
print("Mueller-Muller Symbol Timing Recovery")
print("="*60)

# Adjustable parameters
sps = 16  # samples per symbol
mu_initial = 0.5  # Start at middle of symbol period
loop_gain = 0.05  # Try adjusting this: 0.001, 0.01, 0.05, 0.1

print(f"\nParameters:")
print(f"  Samples per symbol: {sps}")
print(f"  Initial mu: {mu_initial}")
print(f"  Loop gain: {loop_gain}")

# Symbol sync
samples = x
samples_interpolated = resample_poly(samples, 32, 1)
mu = mu_initial
out = np.zeros(len(samples) + 10, dtype=np.complex64)
out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)
i_in = 0
i_out = 2

# Arrays to track loop behavior
mu_log = []
mm_val_log = []
symbol_indices = []

print(f"\nRunning symbol sync...")
while i_out < len(samples) and i_in+32 < len(samples):
    out[i_out] = samples_interpolated[i_in*32 + int(mu*32)]
    out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
    x_val = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
    y_val = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
    mm_val = np.real(y_val - x_val)
    mu += sps + loop_gain*mm_val
    i_in += int(np.floor(mu))
    mu = mu - np.floor(mu)

    # Log for visualization
    mu_log.append(mu)
    mm_val_log.append(mm_val)
    symbol_indices.append(i_out)

    i_out += 1

recovered_symbols = out[2:i_out]
print(f"Recovered {len(recovered_symbols)} symbols")

# Create visualization
fig = plt.figure(figsize=(16, 12))

# 1. Eye Diagram
ax1 = plt.subplot(3, 3, 1)
# Take a section of the interpolated signal
eye_samples = 2000
eye_section = samples_interpolated[:eye_samples*32]
eye_period = 32  # samples per symbol after interpolation
for i in range(0, len(eye_section) - eye_period*2, eye_period):
    ax1.plot(np.real(eye_section[i:i+eye_period*2]), alpha=0.1, color='blue')
ax1.set_title('Eye Diagram (Real part)')
ax1.set_xlabel('Sample')
ax1.grid(True)

# 2. Timing Error (mu)
ax2 = plt.subplot(3, 3, 2)
ax2.plot(mu_log[:5000], linewidth=0.5)
ax2.set_title('Timing Error (mu) over time')
ax2.set_xlabel('Symbol')
ax2.set_ylabel('mu (fractional sample offset)')
ax2.grid(True)

# 3. MM Error Signal
ax3 = plt.subplot(3, 3, 3)
ax3.plot(mm_val_log[:5000], linewidth=0.5)
ax3.set_title('Mueller-Muller Error Signal')
ax3.set_xlabel('Symbol')
ax3.set_ylabel('MM Error')
ax3.grid(True)

# 4. Constellation (first 1000 symbols)
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(np.real(recovered_symbols[:1000]), np.imag(recovered_symbols[:1000]),
            alpha=0.5, s=1)
ax4.set_title('Constellation (first 1000 symbols)')
ax4.set_xlabel('I')
ax4.set_ylabel('Q')
ax4.grid(True)
ax4.axis('equal')

# 5. Histogram of mu
ax5 = plt.subplot(3, 3, 5)
ax5.hist(mu_log, bins=50)
ax5.set_title('Distribution of mu')
ax5.set_xlabel('mu')
ax5.set_ylabel('Count')
ax5.grid(True)

# 6. Real part of symbols
ax6 = plt.subplot(3, 3, 6)
ax6.plot(np.real(recovered_symbols[:500]))
ax6.set_title('Real part of symbols (first 500)')
ax6.set_xlabel('Symbol')
ax6.grid(True)

# 7. Imag part of symbols
ax7 = plt.subplot(3, 3, 7)
ax7.plot(np.imag(recovered_symbols[:500]))
ax7.set_title('Imag part of symbols (first 500)')
ax7.set_xlabel('Symbol')
ax7.grid(True)

# 8. Spectrum of recovered signal
ax8 = plt.subplot(3, 3, 8)
fft_data = np.fft.fft(recovered_symbols[:4096])
freqs = np.fft.fftfreq(4096, 1/(sample_rate/sps))
ax8.plot(freqs[:2048], 20*np.log10(np.abs(fft_data[:2048])))
ax8.set_title('Spectrum of recovered symbols')
ax8.set_xlabel('Frequency (Hz)')
ax8.set_ylabel('Magnitude (dB)')
ax8.grid(True)

# 9. Statistics
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
stats_text = f"""
Symbol Sync Statistics:
━━━━━━━━━━━━━━━━━━━━━━
Recovered symbols: {len(recovered_symbols)}
Expected rate: {sample_rate/sps:.1f} Hz

Mu statistics:
  Mean: {np.mean(mu_log):.4f}
  Std:  {np.std(mu_log):.4f}
  Min:  {np.min(mu_log):.4f}
  Max:  {np.max(mu_log):.4f}

MM Error statistics:
  Mean: {np.mean(mm_val_log):.4f}
  Std:  {np.std(mm_val_log):.4f}

Loop Parameters:
  Samples per symbol: {sps}
  Loop gain: {loop_gain}

Convergence check:
  {'✓ GOOD' if np.std(mu_log[1000:]) < 0.1 else '✗ POOR'}
  (std of mu should be < 0.1)
"""
ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig('rds_timing_debug.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to rds_timing_debug.png")
print(f"\nTo adjust loop gain, edit line 41 in this script")
print(f"Current loop_gain = {loop_gain}")
print(f"Try: 0.001 (slower), 0.01 (current), 0.1 (faster)")
