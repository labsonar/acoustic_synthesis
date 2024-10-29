""" Example of use of scenario module
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.scenario as lps_scenario
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.channel_description as lps_channel

environment = lps_env.Environment.random()
channel_desc = lps_channel.Description.get_default()
sample_frequency = lps_qty.Frequency.khz(16)

scenario = lps_scenario.Scenario(environment = environment,
                                 channel_desc = channel_desc,
                                 temp_dir = './result/propagation')

# sonar = lps_sonar.Sonar(
#         sensors=[
#             lps_sonar.AcousticSensor(sensitivity=lps_qty.Sensitivity.db_v_p_upa(-165)),
#             lps_sonar.AcousticSensor(sensitivity=lps_qty.Sensitivity.db_v_p_upa(-140),
#                                      rel_position=lps_dynamic.Displacement(
#                                          lps_qty.Distance.m(50), lps_qty.Distance.m(100)))
#         ],
#         initial_state=lps_dynamic.State(
#                 position = lps_dynamic.Displacement(
#                         lps_qty.Distance.m(0),
#                         lps_qty.Distance.m(0),
#                         lps_qty.Distance.m(40)))
# )
sonar = lps_sonar.Sonar.hidrofone(
        sensitivity=lps_qty.Sensitivity.db_v_p_upa(-165),
        initial_state=lps_dynamic.State(
                position = lps_dynamic.Displacement(
                        lps_qty.Distance.m(0),
                        lps_qty.Distance.m(0),
                        lps_qty.Distance.m(40)))
)

scenario.add_sonar("main", sonar)

ship1 = lps_scenario.Ship(
                ship_id="Ship_1",
                ship_type=lps_scenario.ShipType.TANKER,
                max_speed=lps_qty.Speed.kt(25),
                draft=lps_qty.Distance.m(5),
                initial_state=lps_dynamic.State(
                        position = lps_dynamic.Displacement(
                                lps_qty.Distance.km(-0.1),
                                lps_qty.Distance.km(-0.2)),
                        velocity = lps_dynamic.Velocity(
                                lps_qty.Speed.kt(5),
                                lps_qty.Speed.kt(5)),
                        acceleration = lps_dynamic.Acceleration(
                                lps_qty.Acceleration.m_s2(0),
                                lps_qty.Acceleration.m_s2(0.02))
                )
        )

scenario.add_ship(ship1)

# scenario.simulate(1024 / sample_frequency, 10)
scenario.simulate(lps_qty.Time.s(1), 120)

scenario.geographic_plot("./result/geographic.png")
scenario.relative_distance_plot("./result/distance.png")

start_time = time.time()
signal = scenario.get_sonar_audio("main", fs=sample_frequency)
end_time = time.time()
print("Get audio time: ", end_time-start_time)

print(signal.shape)

time_axis = np.linspace(0, len(signal) / sample_frequency.get_hz(), num=len(signal))

plt.figure(figsize=(10, 4))
plt.plot(time_axis, np.log10(np.abs(signal)), label="Audio Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Audio Signal in Time Domain")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./result/scenario.png")
plt.close()

wavfile.write("./result/scenario.wav", int(sample_frequency.get_hz()), signal)
