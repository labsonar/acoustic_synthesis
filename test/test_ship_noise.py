import os

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tikz
import scipy.signal as scipy

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_signal
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis
import lps_synthesis.scenario.noise_source as lps_ns
import lps_synthesis.scenario.dynamic as lps_dyn
import lps_synthesis.database.ship as syndb_ship
import lps_synthesis.database.dynamic as syndb_dyn


def _plot_spectrogram(signal, fs, filename: str, n_points):
    f, t, Sxx = scipy.spectrogram(signal, fs=fs.get_hz(), nperseg=n_points, noverlap=n_points/2)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    plt.figure()
    plt.imshow(Sxx_dB.T,
            origin='upper',
            aspect='auto',
            extent=[f[0], f[-1], t[-1], t[0]],
            cmap='jet')
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Tempo (s)")
    plt.colorbar(label="dB")
    plt.tight_layout()

    if filename.endswith('.tex'):
        tikz.save(filename)
    else:
        plt.savefig(filename)
    plt.close()

def main():

    seed = 42
    fs = lps_qty.Frequency.hz(16000)

    step_interval = lps_qty.Time.s(0.2)
    simulation_steps = 150

    output_dir = "./result/test_noise_output"
    os.makedirs(output_dir, exist_ok=True)

    ship_info = syndb_ship.ShipInfo(
        seed=seed,
        major_class="TestClass",
        ship_type=lps_ns.ShipType.BULKER,
        mcr_percent=0.8,
        cruising_speed=lps_qty.Speed.kt(15),
        rotacional_frequency=lps_qty.Frequency.rpm(80),
        length=lps_qty.Distance.m(120),
        draft=lps_qty.Distance.m(6),
        n_blades=5,
        n_shafts=1,
        nb_lower=0,
        nb_medium=0,
        nb_greater=1,
        nb_isolated=0,
        nb_oscillating=0,
        nb_concentrated=0,
    )

    print(f"Generated narrowband configs: {ship_info.narrowband_configs}")
    print(f"Generated narrowband configs: {ship_info.brownian_configs}")

    dynamic = syndb_dyn.SimulationDynamic(
        dynamic_type=syndb_dyn.DynamicType.FIXED_DISTANCE,
        shortest=lps_qty.Distance.m(200),
        approaching=False
    )

    ship = ship_info.make_ship(
        dynamic=dynamic,
        step_interval=step_interval,
        simulation_steps=simulation_steps
    )

    # for i in range(200):
    #     nb = lps_ns.NarrowBandNoise(
    #         frequency=lps_qty.Frequency.khz(0.04 * i),
    #         amp_db_p_upa=150
    #     )
    #     ship.add_source(nb)

    ship.move(step_interval=step_interval, n_steps=simulation_steps)

    compiler = lps_ns.NoiseCompiler(
        noise_containers=[ship],
        fs=fs,
        parallel=False
    )

    n_points = 1024*16

    for i, (signal, _, _) in enumerate(compiler):
        lps_signal.save_normalized_wav(signal,
                                       fs,
                                       os.path.join(output_dir, f"source_{i}.wav"))

        _plot_spectrogram(signal,
                          fs,
                          os.path.join(output_dir, f"source_{i}.png"),
                          n_points)

        lps_bb.plot_psd(filename=os.path.join(output_dir, f"source_{i}_mean.png"),
                        noise=signal,
                        fs=fs,
                        window_size=n_points)

        lps_analysis.SpectralAnalysis.LOFAR.plot(
                filename=os.path.join(output_dir, f"source_{i}_lofar.png"),
                data=signal,
                fs=fs,
                params=lps_analysis.Parameters(n_spectral_pts=n_points),
            )


    print(f"WAV files saved in: {output_dir}")


if __name__ == "__main__":
    main()
