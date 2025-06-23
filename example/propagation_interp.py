""" Example of use of propagation module plotting interpolation of impulse response
"""
import matplotlib.pyplot as plt

import lps_synthesis.propagation.channel as lps_channel

channel = lps_channel.PredefinedChannel.SPHERICAL.get_channel()

distances = [100, 102.5, 105, 107.5, 110]


fig, axes = plt.subplots(len(distances), 1, figsize=(12, 2.5 * (len(distances))), sharex=True)

for i, d in enumerate(distances):
    propag_signal = channel.get_ir().get_h_t(channel.source_depths[0], d)

    axes[i].plot(propag_signal, label=f"{d} m", alpha=0.7)
    axes[i].set_title(f"Propagated signal at {d} m")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid(True)

axes[-1].set_xlabel("Time [s]")

plt.tight_layout()
plt.savefig("./result/propagation_interp.png")
