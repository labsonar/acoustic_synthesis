""" Example of use of scenario module
"""
import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.scenario as lps_scenario


scenario = lps_scenario.Scenario()

# scenario.add_sonar(
#         "main",
#         lps_scenario.Sonar.hidrofone(
#                 time=scenario.start_time,
#                 sensitivity=lps_qty.Sensitivity.db_v_p_upa(-180),
#                 initial_state=lps_dynamic.State(
#                         position = lps_dynamic.Displacement(
#                                 lps_qty.Distance.m(0),
#                                 lps_qty.Distance.m(0),
#                                 lps_qty.Distance.m(0)),
#                         velocity = lps_dynamic.Velocity(
#                                 lps_qty.Speed.m_s(0),
#                                 lps_qty.Speed.m_s(0),
#                                 lps_qty.Speed.m_s(0)),
#                         acceleration = lps_dynamic.Acceleration(
#                                 lps_qty.Acceleration.m_s2(0),
#                                 lps_qty.Acceleration.m_s2(0),
#                                 lps_qty.Acceleration.m_s2(0))
#                 )
#         )
# )
scenario.add_sonar(
        "main",
        lps_scenario.Sonar.hidrofone(
                time=scenario.start_time,
                sensitivity=lps_qty.Sensitivity.db_v_p_upa(-180),
                initial_state=lps_dynamic.State(
                        position = lps_dynamic.Displacement(
                                lps_qty.Distance.m(100),
                                lps_qty.Distance.m(100),
                                lps_qty.Distance.m(50)),
                        velocity = lps_dynamic.Velocity(
                                lps_qty.Speed.m_s(-1),
                                lps_qty.Speed.m_s(-1),
                                lps_qty.Speed.m_s(0)),
                        acceleration = lps_dynamic.Acceleration(
                                lps_qty.Acceleration.m_s2(0),
                                lps_qty.Acceleration.m_s2(0),
                                lps_qty.Acceleration.m_s2(0))
                )
        )
)

scenario.add_ship(
        lps_scenario.Ship(
                ship_id="Ship1",
                ship_type=lps_scenario.ShipType.DREDGER,
                time=scenario.start_time,
                initial_state=lps_dynamic.State(
                        position = lps_dynamic.Displacement(
                                lps_qty.Distance.km(-1),
                                lps_qty.Distance.m(100)),
                        velocity = lps_dynamic.Velocity(
                                lps_qty.Speed.m_s(5),
                                lps_qty.Speed.m_s(0)),
                        acceleration = lps_dynamic.Acceleration(
                                lps_qty.Acceleration.m_s2(0),
                                lps_qty.Acceleration.m_s2(0))
                )
        )
)

scenario.add_ship(
        lps_scenario.Ship(
                ship_id="Ship2",
                ship_type=lps_scenario.ShipType.DREDGER,
                time=scenario.start_time,
                initial_state=lps_dynamic.State(
                        position = lps_dynamic.Displacement(
                                lps_qty.Distance.m(500),
                                lps_qty.Distance.km(-5)),
                        velocity = lps_dynamic.Velocity(
                                lps_qty.Speed.m_s(-10),
                                lps_qty.Speed.m_s(5)),
                        acceleration = lps_dynamic.Acceleration(
                                lps_qty.Acceleration.m_s2(0.005),
                                lps_qty.Acceleration.m_s2(0))
                )
        )
)

scenario.add_ship(
        lps_scenario.Ship(
                ship_id="Ship3",
                ship_type=lps_scenario.ShipType.DREDGER,
                time=scenario.start_time,
                initial_state=lps_dynamic.State(
                        position = lps_dynamic.Displacement(
                                lps_qty.Distance.km(-10),
                                lps_qty.Distance.km(-10)),
                        velocity = lps_dynamic.Velocity(
                                lps_qty.Speed.m_s(5),
                                lps_qty.Speed.m_s(1)),
                        acceleration = lps_dynamic.Acceleration(
                                lps_qty.Acceleration.m_s2(-0.001),
                                lps_qty.Acceleration.m_s2(0.001))
                )
        )
)


scenario.add_ship(
        lps_scenario.Ship(
                ship_id="Ship2",
                ship_type=lps_scenario.ShipType.DREDGER,
                time=scenario.start_time,
                initial_state=lps_dynamic.State(
                        position = lps_dynamic.Displacement(
                                lps_qty.Distance.m(500),
                                lps_qty.Distance.km(-5)),
                        velocity = lps_dynamic.Velocity(
                                lps_qty.Speed.m_s(-10),
                                lps_qty.Speed.m_s(5)),
                        acceleration = lps_dynamic.Acceleration(
                                lps_qty.Acceleration.m_s2(0.005),
                                lps_qty.Acceleration.m_s2(0))
                )
        )
)

times = scenario.simulate(lps_qty.Time.m(1), lps_qty.Time.m(80))

scenario.geographic_plot("./result/geographic.png")
scenario.relative_distance_plot("./result/distance.png")
