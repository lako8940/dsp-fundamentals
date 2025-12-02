import matplotlib.pyplot as plt  # Import plotting library for visualization
import numpy as np  # Import numpy for numerical operations and arrays
from math import sqrt  # Import square root function for signal normalization
import random  # Import random for generating random binary data

# Time vector: 100 samples from 0 to 1 second
t = np.linspace(0,1,100)

# Bit duration: 1 second per symbol
tb = 1

# Carrier frequency: 1 Hz (one complete cycle per second)
fc = 1

# In-phase carrier (I): normalized cosine wave for modulating odd bits
# Amplitude sqrt(2/tb) ensures unit energy per symbol period
c1 = sqrt(2/tb)*np.cos(2*np.pi*fc*t)

# Quadrature carrier (Q): normalized sine wave for modulating even bits
# 90 degrees out of phase with cosine for orthogonal signaling
c2 = sqrt(2/tb)*np.sin(2*np.pi*fc*t)

# Generate 16 random bits to transmit
m = []  # Initialize empty list for message bits

# Time window variables for each symbol period
t1 = 0  # Start time of current symbol
t2 = tb  # End time of current symbol

# Generate 16 random binary values (0 or 1 after thresholding at 0.5)
for i in range(16):
    m.append(random.uniform(0,1))   # Generate random float between 0 and 1
    print(m[i])  # Display the random value (will be converted to binary later)


## QPSK MODULATION
# QPSK encodes 2 bits per symbol: one bit on I (in-phase), one bit on Q (quadrature)
# 16 bits total = 8 QPSK symbols

# Storage for I-channel (in-phase) component: 8 symbols x 100 samples each
odd_sig = np.zeros((8,100))

# Storage for Q-channel (quadrature) component: 8 symbols x 100 samples each
even_sig = np.zeros((8,100))

# Index to track which symbol period we're encoding (0 through 7)
symbol_index = 0

# Process all 16 bits in pairs (bits 0-1, 2-3, 4-5, ... 14-15)
for i in range(0,16,2):
    # Create time vector for this specific symbol period
    t = np.linspace(t1,t2,100)

    # Process odd-indexed bit (bit i) for I-channel
    if (m[i]>0.5):  # If random value > 0.5, treat as binary '1'
        m[i] = 1  # Convert to binary 1
        m_s = np.ones((1,len(t)))  # Create array of +1 values (BPSK: +1 for bit '1')
    else:  # If random value <= 0.5, treat as binary '0'
        m[i] = 0  # Convert to binary 0
        m_s = (-1)*np.ones((1,len(t)))  # Create array of -1 values (BPSK: -1 for bit '0')

    # Modulate I-channel: multiply cosine carrier by +1 or -1 (BPSK on I)
    odd_sig[symbol_index,:] = c1*m_s

    # Process even-indexed bit (bit i+1) for Q-channel
    if (m[i+1]>0.5):  # If random value > 0.5, treat as binary '1'
        m[i+1] = 1  # Convert to binary 1
        m_s = np.ones((1,len(t)))  # Create array of +1 values (BPSK: +1 for bit '1')
    else:  # If random value <= 0.5, treat as binary '0'
        m[i+1] = 0  # Convert to binary 0
        m_s = (-1)*np.ones((1,len(t)))  # Create array of -1 values (BPSK: -1 for bit '0')

    # Modulate Q-channel: multiply sine carrier by +1 or -1 (BPSK on Q)
    even_sig[symbol_index,:] = c2*m_s

    # Move to next symbol slot
    symbol_index += 1

    # Advance time window for next symbol (with small 0.01s gap for visualization)
    t1 = t1 + (tb+0.01)
    t2 = t2 + (tb+0.01)

# Combine I and Q channels to create final QPSK signal
# Each symbol is the sum of modulated cosine and modulated sine
qpsk = odd_sig + even_sig

## ADDITIVE WHITE GAUSSIAN NOISE (AWGN) CHANNEL
# Simulate realistic transmission by adding noise to the signal

# Generate Gaussian noise: mean=0, standard deviation=0.1
# Shape [8,100] matches QPSK signal dimensions (8 symbols x 100 samples)
noise = np.random.normal(0, 0.1, [8,100])

# Create received signal: transmitted QPSK signal + noise
# This simulates what a receiver would see after transmission through a noisy channel
channel = noise + qpsk

## PLOTTING THE RECEIVED SIGNAL

# Create a new figure and axis for plotting
fig, ax = plt.subplots()

# Flatten 2D array (8 x 100) into 1D array (800 samples) for continuous waveform
# This concatenates all 8 symbol periods into one long signal
continuous_signal = channel.flatten()

# Create corresponding time axis: 800 samples divided by 100 samples/second = 8 seconds
time_continuous = np.arange(len(continuous_signal)) / 100

# Plot the continuous received signal (QPSK + noise) vs time
ax.plot(time_continuous, continuous_signal)

# Add grid for easier reading of values
ax.grid()

# Label x-axis
ax.set_xlabel('Time')

# Label y-axis
ax.set_ylabel('Amplitude')

# Add descriptive title
plt.title('QPSK Modulated Signal with Noise')

# Display the plot window
plt.show()
