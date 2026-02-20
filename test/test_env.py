""" Simple script for eva test
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as scipy

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig
import lps_synthesis.scenario as lps_scenario
import lps_synthesis.environment.acoustic_site as lps_as
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.database as lps_db
import lps_synthesis.propagation.channel as lps_propag
import lps_synthesis.propagation.models as lps_models


def main():
    """main function of test noise_source."""

    base_dir = "./result/test_env"
    os.makedirs(base_dir, exist_ok=True)

    sample_frequency = lps_qty.Frequency.khz(16)

    simulation_time = lps_qty.Time.s(60)

    environment = lps_env.Environment(rain_value=lps_env.Rain.NONE,
                                    sea_value=lps_env.Sea.STATE_2,
                                    shipping_value=lps_env.Shipping.LEVEL_1,
                                    global_attenuation_db=20)

    sensor = lps_scenario.AcousticSensor(
            sensitivity=lps_scenario.FlatBand(lps_qty.Sensitivity.db_v_p_upa(-180))
        )
    signal_conditioner = lps_scenario.IdealAmplifier(60)
    adc = lps_scenario.ADConverter()


    for seed in range(10):
        environment = lps_env.Environment.random(seed=seed)
        environment.save_plot(os.path.join(base_dir, f"{seed}.png"))

        noise = environment.generate_bg_noise(
            n_samples=int(sample_frequency * simulation_time),
            fs=int(sample_frequency.get_hz())
        )

        noise = sensor.transduce(input_data=noise,
                                    noise_source=None,
                                    fs=sample_frequency)

        noise = signal_conditioner.convert(noise, fs = sample_frequency)

        noise = adc.apply(noise)

        lps_sig.save_wav(noise, sample_frequency, os.path.join(base_dir, f"{seed}.wav"))


if __name__ == "__main__":
    main()
