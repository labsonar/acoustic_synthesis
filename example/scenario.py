""" Example of use of scenario module
"""
import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.scenario as lps_scenario
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.acoustical_channel as lps_channel

environment = lps_env.Environment.random()
channel_desc = lps_channel.Description.get_default()
sample_frequency = lps_qty.Frequency.khz(16)

scenario = lps_scenario.Scenario(environment=environment, channel_desc=channel_desc)

scenario.add_sonar(
        "main",
        lps_sonar.Sonar.hidrofone(sensitivity=lps_qty.Sensitivity.db_v_p_upa(-165))
)

ship1 = lps_scenario.Ship(
                ship_id="Ship_1",
                ship_type=lps_scenario.ShipType.DREDGER,
                initial_state=lps_dynamic.State(
                        position = lps_dynamic.Displacement(
                                lps_qty.Distance.km(-1),
                                lps_qty.Distance.m(100)),
                        velocity = lps_dynamic.Velocity(
                                lps_qty.Speed.m_s(5),
                                lps_qty.Speed.m_s(0))
                )
        )

scenario.add_ship(ship1)

times = scenario.simulate(lps_qty.Time.s(1), lps_qty.Time.m(0.5))

scenario.geographic_plot("./result/geographic.png")
scenario.relative_distance_plot("./result/distance.png")

ship1.generate_noise(fs=sample_frequency)
