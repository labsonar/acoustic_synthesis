""" Example of use of propagation module plotting TRL
"""
import numpy as np

import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.channel as lps_channel

sample_frequency = lps_qty.Frequency.khz(16)

def _get_sin(freq: lps_qty.Frequency):
    fs = lps_qty.Frequency.khz(16)
    ts = 1 / fs.get_hz()
    duracao = 20
    n_samples = int(fs.get_hz() * duracao)
    ret_t = np.arange(0, n_samples * ts, ts)
    ret_signal = np.sin(2 * np.pi * freq.get_hz() * ret_t)
    return ret_t, ret_signal

channel = lps_channel.PredefinedChannel.SPHERICAL.get_channel()
# channel = lps_channel.PredefinedChannel.CYLINDRICAL.get_channel()

t, input_signal = _get_sin(lps_qty.Frequency.khz(4))

ranges = list(range(10,1000,10))
trl = []

for d in ranges:

    r_t = [lps_qty.Distance.m(d) for _ in range(len(input_signal))]
    propag_signal = channel.propagate(input_signal, channel.source_depths[0], r_t)
    trl.append(np.max(propag_signal)/np.max(input_signal))


plt.plot(ranges, -20*np.log10(trl))
plt.grid(True)
plt.tight_layout()
plt.savefig("./result/trl.png")
