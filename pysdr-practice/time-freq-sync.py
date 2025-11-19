import numpy as np  # NumPy for numerical operations and array manipulation
import matplotlib.pyplot as plt  # Matplotlib for visualizing signals (unused but available)
from scipy import signal  # SciPy signal processing functions (unused but available)
import math  # Standard math library (unused but available)

# =============================================================================
# PART 1: BPSK SYMBOL GENERATION
# Generate a baseband BPSK (Binary Phase Shift Keying) signal
# =============================================================================

num_symbols = 100  # Number of data symbols to transmit (each symbol carries 1 bit in BPSK)
sps = 8  # Samples per symbol - oversampling factor for pulse shaping (higher = smoother signal)

# Generate random binary data: array of 100 random bits (0s and 1s)
# This represents the information we want to transmit
bits = np.random.randint(0, 2, num_symbols)

pulse_train = np.array([])  # Initialize empty array to build our pulse train

# Convert bits to BPSK symbols and create pulse train with zero-padding
for bit in bits:
    pulse = np.zeros(sps)  # Create array of 8 zeros (one symbol period worth of samples)
    pulse[0] = bit*2-1  # Map bit to BPSK: 0 -> -1, 1 -> +1 (antipodal signaling for better noise immunity)
    pulse_train = np.concatenate((pulse_train, pulse))  # Append this symbol's samples to the signal

# =============================================================================
# PART 2: RAISED-COSINE PULSE SHAPING FILTER
# Shapes pulses to minimize inter-symbol interference (ISI) while limiting bandwidth
# =============================================================================

num_taps = 101  # Filter length - odd number centers the filter, more taps = sharper rolloff
beta = 0.35  # Roll-off factor (0-1): controls bandwidth vs time-domain spread tradeoff
              # beta=0 is ideal sinc (min BW, max ISI), beta=1 is widest BW but fastest decay
Ts = sps  # Symbol period in samples (8 samples = 1 symbol period at normalized rate)

# Create time indices centered at 0 for symmetric filter impulse response
# Range: -51 to +51 (101 taps total), centered for zero-phase response
t = np.arange(-51, 52)

# Raised-cosine filter impulse response formula:
# h(t) = sinc(t/Ts) * cos(pi*beta*t/Ts) / (1 - (2*beta*t/Ts)^2)
# - sinc(t/Ts): ideal Nyquist filter (zero crossings at symbol intervals)
# - cos term: spectral shaping for smoother rolloff
# - denominator: compensates for cos term to maintain Nyquist criterion
h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

# Apply pulse shaping by convolving impulse train with raised-cosine filter
# This spreads each impulse into a smooth pulse that doesn't interfere with neighbors
samples = np.convolve(pulse_train, h)

# =============================================================================
# PART 3: FRACTIONAL DELAY FILTER
# Simulates timing offset that occurs in real systems (receiver clock not aligned)
# =============================================================================

delay = 0.4  # Fractional sample delay to simulate (0.4 samples = timing misalignment)
N = 21  # Number of filter taps - odd for symmetric filter, more taps = better accuracy

# Create symmetric tap indices centered at 0: [-10, -9, ..., 0, ..., 9, 10]
n = np.arange(-(N-1)//2, N//2+1)

# Fractional delay filter using shifted sinc function
# sinc(n - delay) creates an ideal interpolation filter shifted by 'delay' samples
# This effectively "moves" the signal by a fractional sample amount
h = np.sinc(n - delay)

# Apply Hamming window to truncated sinc to reduce spectral leakage
# Without windowing, abrupt truncation causes ripples in frequency response
h *= np.hamming(N)

# Normalize filter to unity gain (sum of taps = 1)
# Prevents amplitude change when filtering
h /= np.sum(h)

# Apply fractional delay filter to simulate receiver timing offset
samples = np.convolve(samples, h)

# =============================================================================
# PART 4: FREQUENCY OFFSET SIMULATION
# Simulates carrier frequency offset between transmitter and receiver oscillators
# =============================================================================

fs = 1e6  # Sample rate: 1 MHz (1 million samples per second)
fo = 13000  # Frequency offset: 13 kHz difference between TX and RX local oscillators
Ts = 1/fs  # Sample period: 1 microsecond (time between consecutive samples)

# Create time vector: one time value per sample, starting at 0
# Length matches signal length, spacing = sample period
t = np.arange(0, Ts*len(samples), Ts)

# Apply frequency offset by multiplying with complex exponential
# e^(j*2*pi*fo*t) rotates the signal in complex plane at rate fo
# This simulates the phase rotation caused by LO frequency mismatch
# Result: signal becomes complex (even if it was real before)
samples = samples * np.exp(1j*2*np.pi*fo*t)

# =============================================================================
# PART 5: MUELLER-MULLER TIMING RECOVERY
# Digital timing synchronization algorithm to find optimal sampling instants
# Corrects for the fractional delay and any timing drift
# =============================================================================

mu = 0  # Fractional interpolation index (0 to 1): tracks sub-sample timing estimate
        # Represents how far between integer samples the optimal point is

# Pre-allocate output arrays (slightly larger than input to avoid index errors)
out = np.zeros(len(samples) + 10, dtype=np.complex64)  # Recovered symbols
out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)  # Decision-directed values (hard decisions)

i_in = 0  # Index into input samples array (advances by ~sps each iteration)
i_out = 2  # Index into output array (starts at 2 because algorithm needs 2 previous values)

# Main timing recovery loop - processes input samples to find optimal sampling points
while i_out < len(samples) and i_in+16 < len(samples):
    # Sample the signal at current estimated optimal point
    # In practice, would use interpolation; here we use nearest integer sample
    out[i_out] = samples[i_in]

    # Make hard decisions (slice to constellation points) for timing error detector
    # For BPSK: real part > 0 maps to 1, else 0; same for imaginary
    # These "railed" values represent ideal constellation points
    out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)

    # Mueller-Muller timing error detector (TED) calculation
    # Uses current, previous, and 2-samples-ago values to estimate timing error
    # x term: early-late comparison using decisions
    x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
    # y term: early-late comparison using actual samples
    y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])

    # Timing error: difference between y and x gives timing correction direction
    # Positive = sample too early, Negative = sample too late
    mm_val = np.real(y - x)

    # Update timing estimate using proportional control (loop filter)
    # sps: nominal samples to advance (one symbol period)
    # 0.3*mm_val: correction term (0.3 is loop gain - tradeoff between speed and stability)
    mu += sps + 0.3*mm_val

    # Calculate integer part of mu - this is how many input samples to skip
    i_in += int(np.floor(mu))

    # Keep only fractional part of mu for next iteration
    # This tracks the sub-sample timing offset
    mu = mu - np.floor(mu)

    # Move to next output symbol
    i_out += 1

# Trim output array: remove initial zeros and unfilled tail
# First 2 samples were placeholders; anything after i_out was never written
out = out[2:i_out]

# Store recovered symbols back to samples for further processing
# (e.g., frequency offset correction with Costas Loop)
samples = out