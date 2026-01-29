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

    base_dir = "./result/eva"
    os.makedirs(base_dir, exist_ok=True)

    sample_frequency = lps_qty.Frequency.khz(16)

    acoustic_scenario = lps_db.AcousticScenario(lps_db.Location.GUANABARA_BAY,
                                                lps_as.Season.SPRING)
    dynamic = lps_db.SimulationDynamic(lps_db.DynamicType.CPA_IN, lps_qty.Distance.m(15))
    ship_speed = lps_qty.Speed.kt(15)
    simulation_time = lps_qty.Time.s(60)
    simulation_step=lps_qty.Time.s(0.5)
    zoom_samples = int(lps_qty.Time.s(0.5) * sample_frequency)

    channel = acoustic_scenario.get_channel(lps_models.Oases())
    # channel = lps_propag.PredefinedChannel.SPHERICAL.get_channel(lps_models.Oases())
    # environment = acoustic_scenario.get_env()
    environment = lps_env.Environment(rain_value=lps_env.Rain.NONE,
                                    sea_value=lps_env.Sea.STATE_2,
                                    shipping_value=lps_env.Shipping.LEVEL_1,
                                    global_attenuation_dB=20)
    environment = None

    ship_is = dynamic.get_ship_initial_state(speed=ship_speed, interval=simulation_time)

    ship1 = lps_scenario.Ship(
            ship_id="Ship_1",
            propulsion=lps_scenario.CavitationNoise(
                ship_type=lps_scenario.ShipType.BULKER,
                cruise_rotacional_frequency = lps_qty.Frequency.rpm(80),
            ),
            initial_state=ship_is
        )

    # ship1.add_source(lps_scenario.NarrowBandNoise(
    #         frequency=lps_qty.Frequency.khz(4),
    #         amp_db_p_upa=80)
    #     )

    sonar_is = dynamic.get_sonar_initial_state(ship=ship1)

    sonar = lps_scenario.Sonar.planar(
            n_staves = 5,
            spacing = lps_qty.Distance.m(0.085),
            sensitivity = lps_qty.Sensitivity.db_v_p_upa(-200),
            initial_state=sonar_is
    )

    scenario = lps_scenario.Scenario(channel = channel,
                                    environment = environment,
                                    step_interval= simulation_step)
    scenario.add_sonar("main", sonar)
    scenario.add_noise_container(ship1)

    scenario.simulate(int(simulation_time/simulation_step))

    scenario.geographic_plot(os.path.join(base_dir,"geographic.png"))
    scenario.relative_distance_plot(os.path.join(base_dir,"distance.png"))
    scenario.velocity_plot(os.path.join(base_dir,"velocity.png"))
    scenario.relative_velocity_plot(os.path.join(base_dir,"relative_velocity.png"))

    start_time = time.time()
    signal = scenario.get_sonar_audio("main", fs=sample_frequency)
    end_time = time.time()
    print("Get audio time: ", end_time-start_time)

    time_axis = np.linspace(0, len(signal) / sample_frequency.get_hz(), num=len(signal))

    peak_idx = np.argmax(np.abs(signal))
    half = zoom_samples // 2
    i0 = max(0, peak_idx - half)
    i1 = min(len(signal), peak_idx + half)

    plt.figure(figsize=(10, 4))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, signal)
    plt.ylabel("Amplitude")
    plt.title("Audio Signal - Full")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_axis[i0:i1], signal[i0:i1])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Zoom around peak (sample {peak_idx})")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,"scenario_time.png"))
    plt.close()

    wavfile.write(os.path.join(base_dir,"scenario.wav"), int(sample_frequency.get_hz()), signal)

    float_signal = signal.astype(np.float32) / (2**15 -1)
    f, t, S = scipy.spectrogram(float_signal[:,0], fs=sample_frequency.get_hz(), nperseg=2048)

    plt.figure(figsize=(10, 6))
    # plt.imshow(20 * np.log10(np.abs(np.clip(S, 1e-10, None))),
    #            interpolation='none', aspect='auto',
    #            extent=[t.min(), t.max(), f.max(), f.min()])

    plt.pcolormesh(t, f, 20 * np.log10(np.clip(S, 1e-10, None)), shading='gouraud')
    plt.ylabel('Frequência [Hz]')
    plt.xlabel('Tempo [s]')
    plt.colorbar(label='Intensidade [dB]')
    plt.savefig(os.path.join(base_dir,"scenario_spectro.png"))
    plt.close()

if __name__ == "__main__":
    main()
