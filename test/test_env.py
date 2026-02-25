""" Simple script for eva test
"""
import os
import time
import tqdm
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
import lps_synthesis.scenario.noise_source as lps_ns
import lps_synthesis.scenario.dynamic as lps_dyn
import lps_synthesis.database.ship as syndb_ship
import lps_synthesis.database.dynamic as syndb_dyn
import lps_sp.acoustical.broadband as lps_bb


def rms(x):
    return np.sqrt(np.mean(x**2))

def save(input_noise, name, sensor, signal_conditioner, adc, sample_frequency, base_dir):
    noise = sensor.transduce(input_data=input_noise,
                                noise_source=None,
                                fs=sample_frequency)


    noise = signal_conditioner.convert(noise, fs = sample_frequency)

    noise = adc.apply(noise)

    lps_sig.save_wav(noise, sample_frequency, os.path.join(base_dir, f"{name}.wav"))

    print(f"### {name} ###")
    print(f"Média: {np.mean(input_noise):.4f} µPa")
    print(f"Variância: {np.var(input_noise):.4e} µPa²")
    print(f"RMS: {rms(input_noise):.4f} µPa")
    print(f"SPL (RMS): {20*np.log10(rms(input_noise)):.2f} dB re 1µPa @1m")
    print("")

def pamguide_test():

    base_dir = "./result/test_env"
    os.makedirs(base_dir, exist_ok=True)

    sample_frequency = lps_qty.Frequency.khz(48)

    simulation_time = lps_qty.Time.s(10)
    step_interval = lps_qty.Time.s(0.2)
    simulation_steps = int(simulation_time/step_interval)
    n_samples=int(sample_frequency * simulation_time)

    environment = lps_env.Environment(rain_value=lps_env.Rain.NONE,
                                    sea_value=lps_env.Sea.STATE_2,
                                    shipping_value=lps_env.Shipping.LEVEL_1,
                                    global_attenuation_db=20)

    sensor = lps_scenario.AcousticSensor(
            sensitivity=lps_scenario.FlatBand(lps_qty.Sensitivity.db_v_p_upa(-200))
        )
    signal_conditioner = lps_scenario.IdealAmplifier(0)
    adc = lps_scenario.ADConverter([-2, 2])

    print("")
    print("### broadband ###")
    print("")

    white_noise, _ = lps_bb.generate(
            frequencies = np.array([800 * i for i in range(10)]),
            psd_db = np.array([153.2 for i in range(10)]),
            n_samples=n_samples,
            fs = sample_frequency.get_hz())

    # max_psd = 153.2 # dB ref 1μPa/√Hz @1m

    # rng = np.random.default_rng(seed = 42)
    # psd_linear = 10 ** (max_psd / 20)
    # sigma = psd_linear * np.sqrt(sample_frequency.get_hz()/2)
    # white_noise = rng.normal(0, sigma, n_samples)
    save(white_noise, "white", sensor, signal_conditioner, adc, sample_frequency, base_dir)


    gray_noise, _ = lps_bb.generate(
            frequencies = np.array([800 * i for i in range(10)]),
            psd_db = np.array([153.2 if i < 5 else 100 for i in range(10)]),
            n_samples=n_samples,
            fs = sample_frequency.get_hz())
    save(gray_noise, "gray", sensor, signal_conditioner, adc, sample_frequency, base_dir)


    frequencies = np.logspace(np.log10(0.1), np.log10(10000), 100)

    pink_noise, _ = lps_bb.generate(
            frequencies = frequencies,
            psd_db = np.array([153.2 - 10*np.log10(f/10) for f in frequencies]),
            n_samples=n_samples,
            fs = sample_frequency.get_hz())
    save(pink_noise, "pink", sensor, signal_conditioner, adc, sample_frequency, base_dir)

    print("")
    print("### narrowband ###")
    print("")

    power_linear = 10 ** (197 / 10)
    A = np.sqrt(2 * power_linear)

    t = np.arange(n_samples) / sample_frequency.get_hz()
    sin = A * np.sin(2 * np.pi * 1e3 * t)
    save(sin, "sin", sensor, signal_conditioner, adc, sample_frequency, base_dir)


    nb = lps_ns.NarrowBandNoise(frequency=lps_qty.Frequency.khz(1),
                           amp_db_p_upa=197)

    element = lps_dyn.Element()
    nb.set_base_element(element)
    element.move(step_interval=step_interval, n_steps=simulation_steps)

    nb_noise = nb.generate_noise(fs=sample_frequency)
    save(nb_noise, "nb_sin", sensor, signal_conditioner, adc, sample_frequency, base_dir)

    sum = nb_noise + gray_noise
    save(sum, "sum", sensor, signal_conditioner, adc, sample_frequency, base_dir)

def env_test():

    base_dir = "./result/test_env"
    os.makedirs(base_dir, exist_ok=True)

    sample_frequency = lps_qty.Frequency.khz(16)

    simulation_time = lps_qty.Time.s(30)
    step_interval = lps_qty.Time.s(0.2)
    simulation_steps = int(simulation_time/step_interval)
    n_samples=int(sample_frequency * simulation_time)

    environment = lps_env.Environment(rain_value=lps_env.Rain.NONE,
                                    sea_value=lps_env.Sea.STATE_2,
                                    shipping_value=lps_env.Shipping.LEVEL_1,
                                    global_attenuation_db=0)

    sensor = lps_scenario.AcousticSensor(
            sensitivity=lps_scenario.FlatBand(lps_qty.Sensitivity.db_v_p_upa(-180))
        )
    signal_conditioner = lps_scenario.IdealAmplifier(40)
    adc = lps_scenario.ADConverter([-5, 5])

    print("")
    print("### env simulator ###")
    print("")


    for seed in range(10):
        environment = lps_env.Environment.random(seed=seed)
        environment.save_plot(os.path.join(base_dir, f"{seed}.png"))

        noise = environment.generate_bg_noise(
            n_samples=n_samples,
            fs=int(sample_frequency.get_hz())
        )


        save(noise, f"{seed}", sensor, signal_conditioner, adc, sample_frequency, base_dir)

        break

    compiled_signal = []
    filter_state = None

    rng = np.random.default_rng(seed = 0)
    environment = lps_env.Environment.random(seed=0)
    freq, psd = environment.to_psd()

    step_samples = int(sample_frequency * step_interval)
    for _ in tqdm.tqdm(range(simulation_steps), desc="noise", leave=False):
        noise, filter_state = lps_bb.generate(frequencies = freq,
                                   psd_db = psd,
                                   n_samples = step_samples,
                                   fs = sample_frequency.get_hz(),
                                   filter_state = filter_state,
                                   seed = rng)

        compiled_signal.append(noise)

    compiled_signal = np.concatenate(compiled_signal)
    save(compiled_signal, "env", sensor, signal_conditioner, adc, sample_frequency, base_dir)


def ship_test():

    base_dir = "./result/test_env"
    os.makedirs(base_dir, exist_ok=True)

    sample_frequency = lps_qty.Frequency.khz(16)

    simulation_time = lps_qty.Time.s(30)
    step_interval = lps_qty.Time.s(0.2)
    simulation_steps = int(simulation_time/step_interval)
    n_samples=int(sample_frequency * simulation_time)

    sensor = lps_scenario.AcousticSensor(
            sensitivity=lps_scenario.FlatBand(lps_qty.Sensitivity.db_v_p_upa(-180))
        )
    signal_conditioner = lps_scenario.IdealAmplifier(0)
    adc = lps_scenario.ADConverter([-5, 5])

    print("")
    print("### ship simulator ###")
    print("")

    for type in [lps_ns.ShipType.BULKER, lps_ns.ShipType.FISHING]:

        ship_info = syndb_ship.ShipInfo(
            seed=42,
            major_class="TestClass",
            ship_type=type,
            mcr_percent=0.8,
            cruising_speed=lps_qty.Speed.kt(15),
            rotacional_frequency=lps_qty.Frequency.rpm(80),
            length=lps_qty.Distance.m(120),
            draft=lps_qty.Distance.m(6),
            n_blades=5,
            n_shafts=1,
            nb_lower=1,
            nb_medium=1,
            nb_greater=1,
            nb_isolated=1,
            nb_oscillating=1,
            nb_concentrated=1,
        )

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

        ship.move(step_interval=step_interval, n_steps=simulation_steps)

        compiler = lps_ns.NoiseCompiler(
            noise_containers=[ship],
            fs=sample_frequency,
            parallel=False
        )

        print(ship_info.narrowband_configs)

        for signal, _, _ in compiler:

            save(signal, str(type), sensor, signal_conditioner, adc, sample_frequency, base_dir)

            freqs, intensity = ship_info.ship_type.to_psd(fs=sample_frequency,
                                    lenght=ship_info.length,
                                    speed=ship[0].velocity.get_magnitude_xy())

            freqs = [f.get_hz() for f in freqs]
            plt.figure(figsize=(10, 6))
            plt.plot(freqs, intensity)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel("Power Spectral Density (dB ref 1 µPa / √Hz)")
            plt.semilogx()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, f"{type}.png"))
            plt.close()

            break

if __name__ == "__main__":
    # pamguide_test()
    # env_test()
    ship_test()
