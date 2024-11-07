""" Example of use of scenario module
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as scipy

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.scenario as lps_scenario
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.channel as lps_channel

lps_channel.TEMP_DEFAULT_DIR = "./result/propagation"

environment = lps_env.Environment(rain_value=lps_env.Rain.HEAVY,
                                  sea_value=lps_env.Sea.STATE_5,
                                  shipping_value=lps_env.Shipping.LEVEL_5)
channel = lps_channel.PredefinedChannel.BASIC.get_channel()
sample_frequency = lps_qty.Frequency.khz(16)

scenario = lps_scenario.Scenario(channel = channel,
                                 environment = environment)

# sonar = lps_sonar.Sonar.planar(
#         n_staves=32,
#         spacing=lps_qty.Distance.m(0.015),
#         sensitivity=lps_qty.Sensitivity.db_v_p_upa(-150),
#         initial_state=lps_dynamic.State(
#                 position = lps_dynamic.Displacement(
#                         lps_qty.Distance.m(0),
#                         lps_qty.Distance.m(0)))
# )
sonar = lps_sonar.Sonar.cylindrical(
        n_staves=32,
        radius=lps_qty.Distance.m(3),
        sensitivity=lps_qty.Sensitivity.db_v_p_upa(-150),
        initial_state=lps_dynamic.State(
                position = lps_dynamic.Displacement(
                        lps_qty.Distance.m(0),
                        lps_qty.Distance.m(0)))
)


scenario.add_sonar("main", sonar)


ship1 = lps_scenario.Ship(
                ship_id="Ship_1",
                propulsion=lps_scenario.CavitationNoise(
                    ship_type=lps_scenario.ShipType.CONTAINERSHIP,
                    n_blades=6,
                    n_shafts=1,
                #     shaft_error=0.1
                ),
                max_speed=lps_qty.Speed.kt(15),
                draft=lps_qty.Distance.m(4),
                initial_state=lps_dynamic.State(
                        position = lps_dynamic.Displacement(
                                lps_qty.Distance.km(-0.5),
                                lps_qty.Distance.km(0.2)),
                        velocity = lps_dynamic.Velocity(
                                lps_qty.Speed.kt(4),
                                lps_qty.Speed.kt(0)),
                        acceleration = lps_dynamic.Acceleration(
                                lps_qty.Acceleration.m_s2(0),
                                lps_qty.Acceleration.m_s2(0))
                )
        )

# ship1.add_source(lps_scenario.Sin(frequency=lps_qty.Frequency.khz(0.5), amp_db_p_upa=95))
# ship1.add_source(lps_scenario.Sin(frequency=lps_qty.Frequency.khz(4), amp_db_p_upa=90))
# ship1.add_source(lps_scenario.Sin(frequency=lps_qty.Frequency.khz(6), amp_db_p_upa=80))

scenario.add_noise_container(ship1)


ship2 = lps_scenario.Ship(
                ship_id="Ship_2",
                propulsion=lps_scenario.CavitationNoise(
                    ship_type=lps_scenario.ShipType.RECREATIONAL,
                    n_blades=4,
                    n_shafts=1,
                #     shaft_error=0.1
                ),
                max_speed=lps_qty.Speed.kt(20),
                draft=lps_qty.Distance.m(15),
                initial_state=lps_dynamic.State(
                        position = lps_dynamic.Displacement(
                                lps_qty.Distance.km(0.3),
                                lps_qty.Distance.km(-0.8)),
                        velocity = lps_dynamic.Velocity(
                                lps_qty.Speed.kt(2),
                                lps_qty.Speed.kt(4)),
                        acceleration = lps_dynamic.Acceleration(
                                lps_qty.Acceleration.m_s2(-0.005),
                                lps_qty.Acceleration.m_s2(0))
                )
        )

# ship2.add_source(lps_scenario.Sin(frequency=lps_qty.Frequency.khz(0.5), amp_db_p_upa=95))
ship2.add_source(lps_scenario.Sin(frequency=lps_qty.Frequency.khz(4), amp_db_p_upa=90))
# ship2.add_source(lps_scenario.Sin(frequency=lps_qty.Frequency.khz(6), amp_db_p_upa=80))

scenario.add_noise_container(ship2)

scenario.simulate(lps_qty.Time.s(1), 10*60)

scenario.geographic_plot("./result/geographic.png")
scenario.relative_distance_plot("./result/distance.png")
scenario.velocity_plot("./result/velocity.png")
scenario.relative_velocity_plot("./result/relative_velocity.png")

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
plt.savefig("./result/scenario_time.png")
plt.close()

wavfile.write("./result/scenario.wav", int(sample_frequency.get_hz()), signal)

float_signal = signal.astype(np.float32) / (2**15 -1)
f, t, S = scipy.spectrogram(float_signal[:,0], fs=sample_frequency.get_hz(), nperseg=2048)

plt.figure(figsize=(10, 6))
# plt.imshow(20 * np.log10(np.abs(np.clip(S, 1e-10, None))), interpolation='none', aspect='auto',
#            extent=[t.min(), t.max(), f.max(), f.min()])
plt.pcolormesh(t, f, 20 * np.log10(np.clip(S, 1e-10, None)), shading='gouraud')
plt.title('Espectrograma do Sinal')
plt.ylabel('Frequência [Hz]')
plt.xlabel('Tempo [s]')
plt.colorbar(label='Intensidade [dB]')
plt.savefig("./result/scenario_spectro.png")
plt.close()
