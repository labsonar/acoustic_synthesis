import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.channel as lps_channel

lps_channel.DEFAULT_DIR = "./result/propagation"

def get_sin(freq: lps_qty.Frequency):
    sample_frequency = lps_qty.Frequency.khz(16)
    ts = 1 / sample_frequency.get_hz()
    duracao = 20
    n_samples = int(sample_frequency.get_hz() * duracao)
    ret_t = np.arange(0, n_samples * ts, ts)
    ret_signal = np.sin(2 * np.pi * freq.get_hz() * ret_t)
    return ret_t, ret_signal

def get_white_noise():
    sample_frequency = lps_qty.Frequency.khz(16)
    ts = 1 / sample_frequency.get_hz()
    duration = 20
    n_samples = int(sample_frequency.get_hz() * duration)
    ret_t = np.arange(0, n_samples * ts, ts)

    noise = np.random.normal(0, 1, size=n_samples)
    rms = np.sqrt(np.mean(noise**2))
    noise /= rms
    return ret_t, noise

sin_frequency = lps_qty.Frequency.hz(500)
channel = lps_channel.PredefinedChannel.DUMMY.get_channel()

channel.get_ir().print_h_t_tau("./result/h_t_tau.png")

# t, input_signal = get_white_noise()
t, input_signal = get_sin(lps_qty.Frequency.khz(1))

rms_values = []
distances = list(range(100, 600, 100))

fig, axes = plt.subplots(len(distances) + 1, 1, figsize=(12, 2.5 * (len(distances) + 1)), sharex=True)

axes[0].plot(t[:250], input_signal[:250], label="Original", color="black", alpha=0.7)
axes[0].set_title("Original signal")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

for i, d in enumerate(distances):
    r_t = [lps_qty.Distance.m(d) for _ in range(len(input_signal))]
    propag_signal = channel.propagate(input_signal, channel.source_depths[0], r_t)

    axes[i + 1].plot(t[:250], propag_signal[:250], label=f"{d} m", alpha=0.7)
    axes[i + 1].set_title(f"Propagated signal at {d} m")
    axes[i + 1].set_ylabel("Amplitude")
    axes[i + 1].grid(True)

    rms = np.sqrt(np.mean(propag_signal**2))
    rms_values.append(rms)

axes[-1].set_xlabel("Time [s]")

plt.tight_layout()
plt.savefig("./result/propagation_evaluation.png")

plt.figure(figsize=(8, 5))
plt.plot(distances, 20*np.log10(rms_values), marker='o')
plt.xlabel("Distance [m]")
plt.ylabel("Gain (dB)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./result/gain_vs_distance.png")
plt.show()