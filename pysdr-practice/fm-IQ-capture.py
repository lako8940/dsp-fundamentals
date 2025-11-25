import numpy as np
from scipy.signal import resample_poly, firwin, bilinear, lfilter
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr

sdr = RtlSdr()
sdr.sample_rate = 250e3 # Hz
sdr.center_freq = 103.2e6   # Hz - Confirmed RDS via GQRX
sdr.freq_correction = 60  # PPM
sdr.gain = 'auto'
sdr.set_bandwidth(120e3)


# Capture 2.5M samples (10 seconds @ 250kHz) for better RDS sync
num_samples = 10

samples = sdr.read_samples(num_samples)

# Save to .IQ file
output_filename = 'fm_capture.iq'
samples.tofile(output_filename)

print(f"\nSaved {len(samples)} IQ samples to {output_filename}")
print(f"File size: {samples.nbytes / 1e6:.2f} MB")

# Clean up
sdr.close()
