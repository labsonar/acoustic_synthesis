import time
import subprocess
import tqdm
import numpy as np


import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation as lps_prop
import lps_synthesis.oases as oases


depths = oases.Sweep(start=lps_qty.Distance.m(5), step=lps_qty.Distance.m(5), n_steps=20)
ranges = oases.Sweep(start=lps_qty.Distance.m(10), step=lps_qty.Distance.m(10), n_steps=40)

filename="./result/test.dat"

start_time = time.time()

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

end_time = time.time()
print(f"Tempo decorrido: {(end_time - start_time):.4f} segundos")

outs_1, sd, z, range_, f, fc, omegim, dt = oases.trf_reader(filename)
print(outs_1.shape)
print('z: ', z)
print('range_: ', range_)



outs_2 = np.zeros((depths.get_n_steps(), ranges.get_n_steps()))

start_time = time.time()
for d, depth in enumerate(tqdm.tqdm(depths.get_all(), desc="Depths", leave = False)):
    for r, current_range in enumerate(tqdm.tqdm(ranges.get_all(), desc="Ranges", leave = False)):

        depth_sweep = oases.Sweep(start=depth, step=depths.get_step(), n_steps=1)
        range_sweep = oases.Sweep(start=current_range, step=ranges.get_step(), n_steps=1)

        ssp = lps_prop.SSP()

        ssp.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))
        # ssp.add(lps_qty.Distance.m(10), lps_qty.Speed.m_s(1400))
        # ssp.add(lps_qty.Distance.m(100), lps_qty.Speed.m_s(1600))

        channel  = lps_prop.AcousticalChannel(ssp, None)
        # channel  = lps_prop.AcousticalChannel(ssp, lps_prop.BottomType.ROCK)

        oases.export_dat_file(channel=channel,
                            frequency=lps_qty.Frequency.hz(150),
                            source_depth = lps_qty.Distance.m(0),
                            sensor_depth = depth_sweep,
                            distance = range_sweep,
                            filename=filename)

        comando = f"oasp {filename[:-4]}"
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True, check=True)

        out, sd, z, range_, f, fc, omegim, dt = oases.trf_reader(filename)
        outs_2[d,r] = np.abs(out[0,0,0])

end_time = time.time()
print(f"Tempo decorrido: {(end_time - start_time):.4f} segundos")

print(outs_2.shape)
print('z: ', depths.get_all())
print('range_: ', ranges.get_all())

plt.imshow(20*np.log10(np.abs(outs_1)), aspect='auto', cmap='jet')
plt.savefig("./result/test1.png")
plt.close()

plt.imshow(20*np.log10(np.abs(outs_2)), aspect='auto', cmap='jet')
plt.savefig("./result/test2.png")
plt.close()
