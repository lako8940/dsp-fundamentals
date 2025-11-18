import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

num_symbols = 10
sps = 8

bits = np.random.randint(0, 2, num_symbols) # Our data to be transmitted, 1's and 0's

x = np.array([])
for bit in bits:
    pulse = np.zeros(sps)
    pulse[0] = bit*2-1 # set the first value to either a 1 or -1
    x = np.concatenate((x, pulse)) # add the 8 samples to the signal

# Create our raised-cosine filter
num_taps = 101
beta = 0.35
Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
t = np.arange(num_taps) - (num_taps-1)//2
h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

# Create subplots to show both plots at the same time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# First plot - pulse train
ax1.plot(x, '.-')
ax1.grid(True)
ax1.set_title('Pulse Train')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Amplitude')

# Second plot - raised-cosine filter
ax2.plot(t, h, '.')
ax2.grid(True)
ax2.set_title('Raised-Cosine Filter')
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Filter our signal, in order to apply the pulse shaping
x_shaped = np.convolve(x, h)
plt.figure(2)
plt.plot(x_shaped, '.-')
for i in range(num_symbols):
    plt.plot([i*sps+num_taps//2,i*sps+num_taps//2], [0, x_shaped[i*sps+num_taps//2]])
plt.grid(True)
plt.show()