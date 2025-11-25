import numpy as np
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

# Read in signal
x = np.fromfile('./fm_capture.iq', dtype=np.complex64)
print(f"Loaded {len(x)} IQ samples")

# Normalize
x = x / np.abs(x).max()

# Quadrature Demod (FM demodulation)
x = 0.5 * np.angle(x[0:-1] * np.conj(x[1:]))
print(f"After FM demod: {len(x)} samples, range: [{x.min():.3f}, {x.max():.3f}]")

# Plot the spectrum to see what we have
plt.figure(figsize=(12, 6))

# Time domain
plt.subplot(2, 1, 1)
plt.plot(x[:10000])
plt.title('FM Demodulated Signal (Time Domain)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# Frequency domain
plt.subplot(2, 1, 2)
freq = np.fft.fftfreq(len(x), 1/250e3)
fft = np.fft.fft(x)
plt.plot(freq[:len(freq)//2] / 1e3, 20*np.log10(np.abs(fft[:len(fft)//2])))
plt.title('FM Demodulated Signal (Frequency Domain)')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude (dB)')
plt.xlim(0, 125)
plt.grid(True)

plt.tight_layout()
plt.savefig('fm_spectrum.png', dpi=150)
print("Saved spectrum plot to fm_spectrum.png")

# Look for RDS pilot at 57 kHz
rds_band_start = int(50e3 / 250e3 * len(x))
rds_band_end = int(64e3 / 250e3 * len(x))
rds_peak = np.max(np.abs(fft[rds_band_start:rds_band_end]))
total_power = np.mean(np.abs(fft)**2)

print(f"\nRDS band (50-64 kHz) peak power: {20*np.log10(rds_peak):.1f} dB")
print(f"Average signal power: {10*np.log10(total_power):.1f} dB")
print(f"\nIf you see a strong peak around 57 kHz in the plot, RDS is present!")
