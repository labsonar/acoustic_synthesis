import time
import subprocess
import tqdm
import numpy as np


import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.acoustical_channel as lps_prop
import lps_synthesis.propagation.oases as oases


filename="./result/test.dat"

ssp = lps_prop.SSP()
ssp.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1534))
# ssp.add(lps_qty.Distance.m(5), lps_qty.Speed.m_s(1536))
# ssp.add(lps_qty.Distance.m(50), lps_qty.Speed.m_s(1536))
# ssp.add(lps_qty.Distance.m(80), lps_qty.Speed.m_s(1530))
# ssp.add(lps_qty.Distance.m(100), lps_qty.Speed.m_s(1528))

ssp.print('./result/ssp.png')


channel  = lps_prop.Description(ssp, lps_prop.BottomType.BASALT, lps_qty.Distance.m(25))

start_time = time.time()

h_f, h_t, ranges, t, frequencies = oases.estimate_transfer_function(
                channel = channel,
                source_depth = lps_qty.Distance.m(5),
                sensor_depth = lps_qty.Distance.m(20),
                max_distance = lps_qty.Distance.m(500),
                distance_points = 100,
                sample_frequency = lps_qty.Frequency.khz(16),
                n_fft = 128,
                filename = filename)


plt.imshow(abs(h_t).T, extent=[
                         0,
                         len(t),
                         ranges[-1].get_m(),
                         ranges[0].get_m()
                        ], aspect='auto', cmap='jet')

plt.savefig('./result/h_t.png')
plt.tight_layout()
plt.close()


plt.imshow(abs(h_f), extent=[ranges[0].get_m(),
                         ranges[-1].get_m(),
                         frequencies[0].get_hz(),
                         frequencies[-1].get_hz()], aspect='auto', cmap='jet')


plt.savefig('./result/h_s.png')
plt.tight_layout()
plt.close()

labels = []
for r_i in range(0, len(ranges), 16):
    plt.plot(t, abs(h_t[:,r_i]))
    labels.append(str(ranges[r_i]))

plt.legend(labels)
plt.savefig('./result/h_t_plot.png')
plt.close()