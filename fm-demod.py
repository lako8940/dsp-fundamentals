import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, firwin, bilinear, lfilter
import matplotlib.pyplot as plt
import math

# Read in signal
import os
file_path = 'pysdr-practice/fm_capture.iq'
file_size = os.path.getsize(file_path)
print(f"File size on disk: {file_size / 1e6:.2f} MB")
x = np.fromfile(file_path, dtype=np.complex64)
sample_rate = 250e3
center_freq = 89.3e6
print(f"Loaded {len(x)} IQ samples from file")

#Low-pass filter 200kHz
pass_band = 200e3
stop_band = 250e3
df = (stop_band - pass_band)/sample_rate
num_taps = math.ceil(4/df)
lpf = firwin(num_taps, pass_band, fs = sample_rate)

# Demodulation
x = np.angle(x[1:]*x[:-1]) 

# De-emphasis filter, H(s) = 1/(RC*s + 1), implemented as IIR via bilinear transform
bz, az = bilinear(1, [50e-6, 1], fs=sample_rate)
audio = lfilter(bz, az, x)

# decimate by 6 to get mono audio
audio = audio[::6]
sample_rate_audio = sample_rate/6

# normalize volume so its between -1 and +1
x /= np.max(np.abs(x))

# some machines want int16s
x *= 32767
x = x.astype(np.int16)

# Save to wav file, you can open this in Audacity for example
wavfile.write('fm.wav', int(sample_rate_audio), x)