""" Simple sonar test. """
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as scipy

import lps_utils.quantities as lps_qty
import lps_sp.acoustical.broadband as lps_bb
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.noise_source as lps_noise
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.channel as lps_channel
import lps_synthesis.scenario.sonar as lps_sonar


def main():
    """ Main function """
    base_dir = "./result/sonar_test"
    os.makedirs(base_dir, exist_ok=True)

    fs = lps_qty.Frequency.khz(16)

    ship = lps_noise.Ship.by_type(lps_noise.ShipType.RECREATIONAL)
    ship.add_source(lps_noise.NarrowBandNoise.with_brownian_modulation(
            frequency=lps_qty.Frequency.khz(4),
            amp_db_p_upa=90,
            amp_std=0.05,
            phase_std=0.2,
            rel_position=lps_dynamic.Displacement(lps_qty.Distance.m(0), lps_qty.Distance.m(0))))

    sonar = lps_sonar.Sonar.planar(
        n_staves=2,
        spacing=lps_qty.Distance.m(2),
        sensitivity = lps_qty.Sensitivity.db_v_p_upa(-180),
        signal_conditioner = lps_sonar.IdealAmplifier(60),
        initial_state = lps_dynamic.State(lps_dynamic.Displacement(lps_qty.Distance.m(0),
                                                                   lps_qty.Distance.m(-5)),
                                            lps_dynamic.Velocity(lps_qty.Speed.kt(0),
                                                                 lps_qty.Speed.kt(0)))
    )

    ship.move(lps_qty.Time.s(1), 15)
    sonar.move(lps_qty.Time.s(1), 15)

    lps_dynamic.Element.plot_trajectories([ship, sonar],
                                          filename=os.path.join(base_dir, "scenario.png"))


    compiler = lps_noise.NoiseCompiler([ship], fs)
    compiler.save_plot(os.path.join(base_dir, "noise_compiler.png"))
    compiler.show_details()
    compiler.save_wavs(base_dir)

    environment = lps_env.Environment.random()
    environment.save_plot(os.path.join(base_dir, "environment.png"))

    channel = lps_channel.PredefinedChannel.CYLINDRICAL.get_channel()


    data = sonar.get_data(noise_compiler=compiler, channel=channel, environment=environment)

    wavfile.write(filename=os.path.join(base_dir, "output.wav"),
                  rate=int(fs.get_hz()),
                  data=data)

    _, axes = plt.subplots(data.shape[1], 1, figsize=(12, 3 * data.shape[1]), sharex=True)

    if data.shape[1] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        freqs, psd_vals = lps_bb.psd(
            signal=data[:, i],
            fs=fs.get_hz())
        ax.plot(freqs, psd_vals)
        ax.set_title(f"Sensor {i} PSD")
        ax.set_ylabel("dB re 1μPa²/Hz")
        ax.grid(True)

    axes[-1].set_xlabel("Frequency [Hz]")

    plt.tight_layout()
    output_path = os.path.join(base_dir, "sonar.png")
    plt.savefig(output_path)
    plt.close()

    f, t, sxx = scipy.spectrogram(data[:, 0], fs=fs.get_hz())

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 20 * np.log10(sxx + 1e-12), shading='gouraud', cmap='viridis')
    plt.title("Spectrogram - Sensor 0")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="dB re 1μPa²/Hz")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "spectrogram.png"))
    plt.close()

if __name__ == "__main__":
    main()
