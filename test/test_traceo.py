import os
import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_utils.subprocess as lps_proc
import lps_synthesis.propagation.channel_description as lps_desc
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.oases as oases
import lps_synthesis.propagation.traceo as traceo

def _main(frequency = lps_qty.Frequency.khz(5)):
    output_dir = "result/traceo"

    os.makedirs(output_dir, exist_ok=True)
    max_depth = lps_qty.Distance.m(50000)

    desc = lps_desc.Description()
    desc.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))
    desc.add(max_depth, lps_qty.Speed.m_s(1500))
    # desc.add(lps_qty.Distance.m(1) + max_depth, lps_layer.BottomType.BASALT)
    desc.remove_air_sea_interface()

    filename = "test.in"
    traceo_input_file = os.path.join(output_dir, filename)
    traceo_output_file = os.path.join(output_dir, "aad.mat")
    traceo.export_file(
            frequency = frequency,
            description = desc,
            source_depths = oases.Sweep(max_depth/2, lps_qty.Distance.m(1), 1),
            sensor_depth = max_depth/2,
            distance = oases.Sweep(lps_qty.Distance.m(10), lps_qty.Distance.m(90), 2),
            filename = traceo_input_file
    )

    if os.path.exists(traceo_output_file):
        os.remove(traceo_output_file)

    lps_proc.run_process(comand=f"traceo {filename}", running_directory=output_dir)

    h, ranges, depths = traceo.read_file(output_dir, frequency)

    h_r = h[:, 0]
    ranges = np.array([int(r.get_m()) for r in ranges])

    TL = -20.0 * np.log10(np.abs(h_r))

    plt.figure()
    plt.semilogx(ranges, TL, marker='o')
    plt.xlabel("Distância (m)")
    plt.ylabel("Perda de Transmissão (dB)")
    plt.title(f"TL vs Distância (z = {depths[0]})")
    plt.grid(True)
    plt.tight_layout()

    filename = f"{int(frequency.get_hz())}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

    for n in range(10):
        d = 10**n
        idx = np.where(ranges == d)[0]

        if idx.size > 0:
            print(f"\tTL({d:.0f} m) = {TL[idx[0]]:.2f} dB")

if __name__ == "__main__":

    for f in np.arange(0.01, 8.01, 0.01):
        freq = lps_qty.Frequency.khz(f)
        print(freq)
        _main(frequency = freq)
