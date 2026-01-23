""" Example of use of scenario module
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as scipy

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario as lps_scenario
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation as lps_propag


def main():
    """main function of test noise_source."""

    base_dir = "./result/scenario"
    os.makedirs(base_dir, exist_ok=True)

    environment = lps_env.Environment(rain_value=lps_env.Rain.LIGHT,
                                    sea_value=lps_env.Sea.STATE_3,
                                    shipping_value=lps_env.Shipping.LEVEL_3)
    channel = lps_channel.PredefinedChannel.CYLINDRICAL.get_channel()
    sample_frequency = lps_qty.Frequency.khz(16)

    scenario = lps_scenario.Scenario(channel = channel,
                                    environment = environment,
                                    step_interval=lps_qty.Time.s(1))

    sonar = lps_sonar.Sonar.hydrophone(
            sensitivity=lps_qty.Sensitivity.db_v_p_upa(-150),
            initial_state=lps_dynamic.State(
                    position = lps_dynamic.Displacement(
                            lps_qty.Distance.m(0),
                            lps_qty.Distance.m(0)))
    )

    scenario.add_sonar("main", sonar)


    ship1 = lps_noise.Ship(
                    ship_id="Ship_1",
                    propulsion=lps_noise.CavitationNoise(
                        ship_type=lps_noise.ShipType.BULKER
                    ),
                    initial_state=lps_dynamic.State(
                            position = lps_dynamic.Displacement(
                                    lps_qty.Distance.km(-0.1),
                                    lps_qty.Distance.km(-0.2)),
                            velocity = lps_dynamic.Velocity(
                                    lps_qty.Speed.kt(10),
                                    lps_qty.Speed.kt(10)),
                            acceleration = lps_dynamic.Acceleration(
                                    lps_qty.Acceleration.m_s2(0),
                                    lps_qty.Acceleration.m_s2(0.02))
                    )
            )

    ship1.add_source(lps_noise.NarrowBandNoise(frequency=lps_qty.Frequency.khz(4),
                                               amp_db_p_upa=80))
    ship1.add_source(lps_noise.NarrowBandNoise(frequency=lps_qty.Frequency.khz(4.05),
                                               amp_db_p_upa=80,
                                               rel_position=lps_dynamic.Displacement(
                                                    lps_qty.Distance.m(50),
                                                    lps_qty.Distance.m(0))))

    scenario.add_noise_container(ship1)

    scenario.simulate(60)

    scenario.geographic_plot(os.path.join(base_dir,"geographic.png"))
    scenario.relative_distance_plot(os.path.join(base_dir,"distance.png"))
    scenario.velocity_plot(os.path.join(base_dir,"velocity.png"))
    scenario.relative_velocity_plot(os.path.join(base_dir,"relative_velocity.png"))

    start_time = time.time()
    signal = scenario.get_sonar_audio("main", fs=sample_frequency)
    end_time = time.time()
    print("Get audio time: ", end_time-start_time)

    time_axis = np.linspace(0, len(signal) / sample_frequency.get_hz(), num=len(signal))

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, signal, label="Audio Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Signal in Time Domain")
    plt.grid(True)
    plt.legend()
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
