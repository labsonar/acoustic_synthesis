import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tikz
import scipy.signal as scipy

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_signal
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis
import lps_synthesis.database.ship as db_ship
import lps_synthesis.database.dynamic as syndb_dyn
import lps_synthesis.scenario.noise_source as lps_ns



def _plot_spectrogram(signal, fs, filename: str):
    f, t, Sxx = scipy.spectrogram(signal, fs=fs.get_hz(), nperseg=4096, noverlap=2048)
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

def _main():

    output_dir = "./result/olocum_ship_noises"
    os.makedirs(output_dir, exist_ok=True)

    catalog_path = "./result/olocum/ship_catalog.csv"
    nb_path = "./result/olocum/ship_catalog_nb_noises.csv"

    catalog = db_ship.ShipCatalog.load(catalog_path)

    catalog.export_narrowband_dataframe(nb_path)

    step_interval = lps_qty.Time.s(0.2)
    simulation_steps = 150
    fs = lps_qty.Frequency.hz(16000)

    dynamic = syndb_dyn.SimulationDynamic(
        dynamic_type=syndb_dyn.DynamicType.FIXED_DISTANCE,
        shortest=lps_qty.Distance.m(1000),
        approaching=False
    )

    for i, ship_info in enumerate(tqdm.tqdm(catalog, desc="Ship", ncols=120, leave=False)):

        if len(ship_info.narrowband_configs) == 0 and len(ship_info.brownian_configs) == 0:
            continue

        print(f"Generated narrowband configs: {ship_info.narrowband_configs}")
        print(f"Generated narrowband configs: {ship_info.brownian_configs}")

        ship = ship_info.make_ship(
            dynamic=dynamic,
            step_interval=step_interval,
            simulation_steps=simulation_steps
        )

        ship.move(step_interval=lps_qty.Time.s(1), n_steps=30)

        compiler = lps_ns.NoiseCompiler(
            noise_containers=[ship],
            fs=fs,
            parallel=False
        )

        for j, (signal, _, _) in enumerate(compiler):
            lps_signal.save_normalized_wav(signal,
                                        fs,
                                        os.path.join(output_dir, f"ship_{i}_{j}.wav"))

            _plot_spectrogram(signal,
                            fs,
                            os.path.join(output_dir, f"ship_{i}_{j}.png"))

            lps_bb.plot_psd(filename=os.path.join(output_dir, f"ship_{i}_{j}_mean.png"),
                            noise=signal,
                            fs=fs)

            lps_analysis.SpectralAnalysis.LOFAR.plot(
                    filename=os.path.join(output_dir, f"ship_{i}_{j}_lofar.png"),
                    data=signal,
                    fs=fs
                )


if __name__ == "__main__":
    _main()
