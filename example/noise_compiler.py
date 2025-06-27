""" Noise Compiler test script """
import os

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.noise_source as lps_noise


def main():
    """ Test main function. """

    base_dir = "./result/noise_sources"
    os.makedirs(base_dir, exist_ok=True)

    fs = lps_qty.Frequency.khz(16)

    x = [0, 0, 1, 0, 0]
    # x = list(range(5))

    noise1 = lps_noise.NarrowBandNoise(
        frequency=lps_qty.Frequency.khz(1),
        amp_db_p_upa=150,
        rel_position = lps_dynamic.Displacement(lps_qty.Distance.m(x[0]), lps_qty.Distance.m(0))
    )

    noise2 = lps_noise.NarrowBandNoise.with_sine_am_modulation(
        frequency=lps_qty.Frequency.khz(2),
        amp_db_p_upa=150,
        am_freq=lps_qty.Frequency.hz(1),
        am_depth=0.3,
        rel_position = lps_dynamic.Displacement(lps_qty.Distance.m(x[1]), lps_qty.Distance.m(0))
    )

    noise3 = lps_noise.NarrowBandNoise.with_sine_fm_modulation(
        frequency=lps_qty.Frequency.khz(3),
        amp_db_p_upa=150,
        oscilation_freq=lps_qty.Frequency.hz(2),
        deviation_freq=lps_qty.Frequency.hz(100),
        rel_position = lps_dynamic.Displacement(lps_qty.Distance.m(x[2]), lps_qty.Distance.m(0)),
    )

    noise4 = lps_noise.NarrowBandNoise.with_brownian_modulation(
        frequency=lps_qty.Frequency.khz(4),
        amp_db_p_upa=150,
        amp_std=0,
        phase_std=0.2,
        rel_position = lps_dynamic.Displacement(lps_qty.Distance.m(x[3]), lps_qty.Distance.m(0))
    )

    noise5 = lps_noise.NarrowBandNoise.with_fm_chirp(
        amp_db_p_upa=150,
        start_frequency=lps_qty.Frequency.khz(5),
        end_frequency=lps_qty.Frequency.khz(6),
        tx_interval=lps_qty.Time.s(1),
        tx_duration=lps_qty.Time.s(0.8),
        rel_position = lps_dynamic.Displacement(lps_qty.Distance.m(x[4]), lps_qty.Distance.m(0))
    )

    container1 = lps_noise.NoiseContainer("Container 1")
    container1.add_source(noise1)
    container1.add_source(noise2)
    container1.add_source(noise3)

    container2 = lps_noise.NoiseContainer("Container 2")
    container2.add_source(noise4)
    container2.add_source(noise5)

    container1.move(lps_qty.Time.s(1), 10)
    container2.move(lps_qty.Time.s(1), 10)

    compiler = lps_noise.NoiseCompiler([container1, container2], fs)
    compiler.save_plot(os.path.join(base_dir, "noise_compiler.png"))
    compiler.show_details()


if __name__ == "__main__":
    main()
