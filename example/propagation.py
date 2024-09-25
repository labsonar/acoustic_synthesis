import time
import subprocess
import tqdm
import numpy as np


import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation as lps_prop
import lps_synthesis.oases as oases


depths = oases.Sweep(start=lps_qty.Distance.m(0), step=lps_qty.Distance.m(25), n_steps=20)
ranges = oases.Sweep(start=lps_qty.Distance.m(10), step=lps_qty.Distance.m(10), n_steps=40)

filename="./result/test.dat"

start_time = time.time()

ssp = lps_prop.SSP()
ssp.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))

for bottom in lps_prop.BottomType:

    channel  = lps_prop.AcousticalChannel(ssp, bottom, lps_qty.Distance.m(400))

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

    plt.imshow(20*np.log10(np.abs(outs_1)),
            extent=[range_[0] * 1e3,
                        range_[-1] * 1e3,
                        z[-1],
                        z[0]],
                aspect='auto', cmap='jet')

    plt.savefig(f"./result/test_{bottom}.png")
    plt.close()
