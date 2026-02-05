# DSP Fundamentals - Portfolio Repo 1

## Project Overview

This is **Repo 1 of 4** in an RF engineering portfolio project demonstrating expertise across antenna design, hardware integration, and signal processing. The portfolio structure:

1. **Repo 1: DSP Fundamentals** (this repo) — Simulation-based DSP demonstrations
2. **Repo 2: Multi-SDR Synchronization** — Clock distribution and inter-channel calibration
3. **Repo 3: Multi-SDR Beamforming (2-element)** — Basic beamforming with real hardware
4. **Repo 4: Four-Element Beamforming Array** — Full system with 868 MHz patch antennas

## Purpose

Demonstrate DSP competency through **cohesive practical projects** rather than isolated function demos. Each notebook tells a complete story with theory, implementation, visualization, and connection to the hardware repos.

## Notebooks

### 1. BPSK Modem (`bpsk_modem.ipynb`) ✅ Complete

End-to-end digital communications system demonstrating:

**Transmitter:**
- Bit generation and BPSK symbol mapping (0→-1, 1→+1)
- Pulse shaping with root-raised cosine (RRC) filter
- Upconversion to 20 kHz carrier

**Channel Model:**
- AWGN at configurable SNR
- Frequency offset (simulates LO mismatch between TX/RX)
- Phase offset (unknown initial phase)

**Receiver:**
- Downconversion to complex baseband
- Matched filtering (RRC)
- Frequency offset estimation via squaring method
- Phase offset estimation via squaring + averaging
- Symbol detection and BER calculation

**Key Visualizations (each RX stage):**
- Time-domain I/Q waveforms
- Power spectral density (Welch's method)
- Constellation diagrams showing: rotating ring → two rotated clusters → aligned on I-axis

**Connection to Beamforming:** The frequency/phase synchronization techniques directly apply to multi-SDR calibration in Repos 2-4.

### 2. Two-Element Direction of Arrival (`doa_simulation.ipynb`) 🔲 Pending

Simulated plane wave arriving at two antennas. Will demonstrate:
- Phase difference extraction from two receivers
- Angle of arrival estimation
- Impact of noise and frequency offset on DoA accuracy
- Direct precursor to hardware implementation in Repo 3

## Technical Parameters (BPSK Modem)

| Parameter | Value |
|-----------|-------|
| Sample rate | 80 kHz |
| Symbol rate | 10 ksym/s |
| Samples per symbol | 8 |
| Carrier frequency | 20 kHz |
| RRC roll-off (α) | 0.35 |
| RRC filter taps | 101 |

## Design Decisions

1. **Welch's method for spectra** — Raw FFT showed noise artifacts; `scipy.signal.welch()` gives clean PSD estimates

2. **Squaring method for sync** — Removes BPSK ±1 modulation, leaving tone at 2× frequency offset. Simple and effective for BPSK.

3. **No BER vs SNR waterfall plot** — Removed to keep focus on the signal processing chain visualization

4. **Consistent constellation sampling** — All constellation plots sample at `samples_per_symbol//2` offsets for symbol centers

## File Structure

```
repo1-dsp-fundamentals/
├── CLAUDE.md                 # This file
├── bpsk_modem.ipynb         # BPSK modem notebook
├── doa_simulation.ipynb     # DoA estimation (pending)
└── README.md                # GitHub-facing readme (pending)
```

## Hardware Context (for reference)

The broader project uses:
- **Antennas:** 868 MHz microstrip inset-fed patch antennas (designed via Balanis formulas, simulated in OpenEMS, fabricated via JLCPCB)
- **Receivers:** 4× RTL-SDRs with shared 28.8 MHz reference clock
- **Clock source:** SI5351 dev board programmed with STM32 Nucleo
- **Test signal source:** HackRF
- **Measurements:** LiteVNA

## Next Steps

1. Create `doa_simulation.ipynb` (Option C from planning)
2. Create GitHub README.md with repo overview
3. Ensure notebooks run cleanly and produce expected outputs
