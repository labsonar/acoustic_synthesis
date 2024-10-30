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
import lps_synthesis.propagation.channel as lps_channel

lps_channel.TEMP_DEFAULT_DIR = "./result/propagation"

environment = lps_env.Environment(rain_value=lps_env.Rain.HEAVY,
                                  sea_value=lps_env.Sea.STATE_4,
                                  shipping_value=lps_env.Shipping.LEVEL_2)
channel = lps_channel.PredefinedChannel.BASIC.get_channel()
sample_frequency = lps_qty.Frequency.khz(16)

scenario = lps_scenario.Scenario(channel = channel,
                                 environment = environment)

sonar = lps_sonar.Sonar.hidrofone(
        sensitivity=lps_qty.Sensitivity.db_v_p_upa(-165),
        initial_state=lps_dynamic.State(
                position = lps_dynamic.Displacement(
                        lps_qty.Distance.m(0),
                        lps_qty.Distance.m(0)))
)

scenario.add_sonar("main", sonar)

ship1 = lps_scenario.Ship(
                ship_id="Ship_1",
                ship_type=lps_scenario.ShipType.BULKER,
                max_speed=lps_qty.Speed.kt(15),
                draft=lps_qty.Distance.m(15),
                propulsion=lps_scenario.Propulsion(
                    ship_type=lps_scenario.ShipType.BULKER,
                    n_blades=5,
                    n_shafts=2,
                    shaft_error=0.1
                ),
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

scenario.add_noise_source(ship1)

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
plt.plot(time_axis, signal, label="Audio Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Audio Signal in Time Domain")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./result/scenario.png")
plt.close()

wavfile.write("./result/scenario.wav", int(sample_frequency.get_hz()), signal)
