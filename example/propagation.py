import subprocess
import tqdm
import numpy as np

import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation as lps_prop
import lps_synthesis.oases as oases

# depths = np.linspace(-200, 200, 21)
# ranges = np.arange(0.5, 0.5 * 41, 0.5)

# outs = np.zeros((len(depths), len(ranges)))

# for d, depth in enumerate(tqdm.tqdm(depths, desc = "Depth", leave=False)):
#     for r, current_range in enumerate(tqdm.tqdm(ranges, desc = "Range", leave=False)):

#         filename="./result/test.dat"

#         ssp = lps_prop.SSP()

#         ssp.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))
#         # ssp.add(lps_qty.Distance.m(10), lps_qty.Speed.m_s(1400))
#         # ssp.add(lps_qty.Distance.m(100), lps_qty.Speed.m_s(1600))

#         channel  = lps_prop.AcousticalChannel(ssp, None)
#         # channel  = lps_prop.AcousticalChannel(ssp, lps_prop.BottomType.ROCK)

#         oases.export_dat_file(channel=channel,
#                             frequency=lps_qty.Frequency.hz(150),
#                             source_depth = lps_qty.Distance.m(0),
#                             sensor_depth = lps_qty.Distance.m(depth),
#                             distance = lps_qty.Distance.m(current_range),
#                             filename=filename)

#         comando = f"oasp {filename[:-4]}"
#         resultado = subprocess.run(comando, shell=True, capture_output=True, text=True, check=True)

#         out, sd, z, range_, f, fc, omegim, dt = oases.trf_reader(filename)
#         outs[d,r] = np.abs(out[0,0,0])

depths = oases.Sweep(start=lps_qty.Distance.m(-200), step=lps_qty.Distance.m(20/3), n_steps=61)
ranges = oases.Sweep(start=lps_qty.Distance.m(0), step=lps_qty.Distance.m(2.5), n_steps=81)

filename="./result/test.dat"

ssp = lps_prop.SSP()
ssp.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))

channel  = lps_prop.AcousticalChannel(ssp, None)

oases.export_dat_file(channel=channel,
                    frequency=lps_qty.Frequency.hz(150),
                    source_depth = lps_qty.Distance.m(0),
                    sensor_depth = depths,
                    distance = ranges,
                    filename=filename)

comando = f"oasp {filename[:-4]}"
resultado = subprocess.run(comando, shell=True, capture_output=True, text=True, check=True)

outs, sd, z, range_, f, fc, omegim, dt = oases.trf_reader(filename)
print(outs.shape)
print(z)
print(range_)
print(outs[0:3,0:3])
print(sd, f, fc, omegim, dt)
outs = np.abs(outs)
outs = 20*np.log10(outs)

plt.imshow(outs, extent=[range_[0] * 1e3,
                         range_[-1] * 1e3,
                         z[-1],
                         z[0]], aspect='auto', cmap='jet')

plt.colorbar(label='Magnitude (dB)')
plt.xlabel('Range (m)')
plt.ylabel('Depth (m)')
plt.title('Magnitude Plot')

plt.savefig("./result/test.png")
plt.close()