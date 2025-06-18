"""Oases test
"""
import matplotlib.pyplot as plt
import numpy as np

import lps_synthesis.propagation.oases as oases

h_t_tau, depths, r, freqs, t = oases.trf_impulse_response_reader("./result/propagation/dummy.trf")

gain = 20*np.log10(np.max(h_t_tau[0,:,:], axis=1))

plt.figure()
for r in range(h_t_tau.shape[1]):
    plt.plot(h_t_tau[0, r, :])
plt.xlabel("Time Index")
plt.ylabel("Amplitude")
plt.title("Impulse responses for all ranges")
plt.grid(True)
plt.savefig("./result/h_t_tau.png")
plt.close()


def build_test_signal(FS, duration_s=5.0, f0=7500, f1=1000, a_pk=1.0, noise_rms=0.2):
    """Generate a two‑tone sine plus AWGN, mirroring the Octave demo."""
    n = int(round(duration_s * FS))
    t_vec = np.arange(n) / FS
    s_sin = a_pk * (np.sin(2 * np.pi * f0 * t_vec) + np.sin(2 * np.pi * f1 * t_vec))
    noise = noise_rms * np.random.randn(n)
    return t_vec, s_sin + noise, s_sin

h_t = h_t_tau[0,0,:]
fs = 16000
times_s = np.array([time.get_s() for time in t])

t_sig, s_in, s_orig = build_test_signal(fs)
s_out = np.convolve(s_in, h_t, mode="same")

n_fft = 1 << int(np.ceil(np.log2(len(s_in))))
f_axis = fs * np.arange(n_fft // 2) / n_fft

s_in_fft_db = 20 * np.log10(np.abs(np.fft.fft(s_in, n_fft))[: n_fft // 2])
s_out_fft_db = 20 * np.log10(np.abs(np.fft.fft(s_out, n_fft))[: n_fft // 2])

plt.figure(figsize=(10, 10))

plt.subplot(3, 2, 1)
plt.plot(t_sig * 1e3, s_orig)
plt.title("Original signal (7.5 kHz + 1 kHz, 1 Vₚ)")
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude [V]")
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(f_axis / 1e3, s_in_fft_db)
plt.title("Spectrum of input signal (|FFT|, dB)")
plt.xlabel("Frequency [kHz]")
plt.ylabel("|S_in(f)| [dB]")
plt.grid(True)
plt.xlim(0, fs / 2000)

plt.subplot(3, 2, 3)
plt.plot(times_s * 1e3, h_t)
plt.title("Channel impulse response h(t)")
plt.xlabel("Time [ms]")
plt.ylabel("h(t)")
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(t_sig * 1e3, s_out, color="k")
plt.title("Output signal (s_out = s_in * h_t)")
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude [V]")
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(f_axis / 1e3, s_out_fft_db, color="m")
plt.title("Spectrum after channel (|FFT|, dB)")
plt.xlabel("Frequency [kHz]")
plt.ylabel("|S_out(f)| [dB]")
plt.grid(True)
plt.xlim(0, fs / 2000)

plt.tight_layout()
plt.savefig("./result/oases.png")