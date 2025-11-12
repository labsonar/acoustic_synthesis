"""
Dynamic Module

Define the dynamics and their implementations for the synthetic dataset.
"""
import enum
import random

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.noise_source as lps_noise
import lps_synthesis.scenario.dynamic as lps_dyn

class DynamicType(enum.Enum):
    """Defines the motion type of the simulated event."""
    FIXED_DISTANCE = enum.auto()
    CPA_IN = enum.auto()
    CPA_OUT = enum.auto()

    def __str__(self):
        return self.name.lower()

class SimulationDynamic:
    """ Describes the dynamic behavior of the acoustic simulation. """

    def __init__(self, dynamic_type: DynamicType, shortest: lps_qty.Distance):
        self.dynamic_type = dynamic_type
        self.shortest = shortest

    def __str__(self):
        return f"{self.dynamic_type.name} (d_min={self.shortest})"

    @staticmethod
    def rand(min_dist: lps_qty.Distance = lps_qty.Distance.m(50),
             max_dist: lps_qty.Distance = lps_qty.Distance.m(250)) -> "SimulationDynamic":
        """Generate a random dynamic configuration."""

        dist = lps_qty.Distance.m(random.randint(min_dist.get_m(), max_dist.get_m()))
        dynamic_type = random.choice(list(DynamicType))
        return SimulationDynamic(dynamic_type, dist)

    def get_ship_initial_state(self, speed: lps_qty.Speed, interval: lps_qty.Time) -> lps_dyn.State:
        """ Define the initial state to fulfill the dynamics. """

        y_offset = self.shortest if self.dynamic_type != DynamicType.CPA_OUT \
                                 else lps_qty.Distance.m(0)

        if self.dynamic_type == DynamicType.FIXED_DISTANCE:
            x_offset = lps_qty.Distance.m(0)
        elif self.dynamic_type == DynamicType.CPA_IN:
            x_offset = -0.5 * speed * interval
        elif self.dynamic_type == DynamicType.CPA_OUT:
            x_offset = (self.shortest + speed * interval) * -1
        else:
            raise UnboundLocalError("get_initial_state not handling {self.dynamic_type} dynamic")

        return lps_dyn.State(
            position = lps_dyn.Displacement(x_offset, y_offset),
            velocity = lps_dyn.Velocity(speed, lps_qty.Speed.kt(0)),
            acceleration = lps_dyn.Acceleration(lps_qty.Acceleration.m_s2(0),
                                                lps_qty.Acceleration.m_s2(0))
        )

    def get_sonar_initial_state(self, ship: lps_noise.Ship) -> lps_dyn.State:
        """ Define the initial state to fulfill the dynamics. """

        if self.dynamic_type == DynamicType.FIXED_DISTANCE:
            sonar_speed = ship.ref_state.velocity.x
        else:
            sonar_speed = lps_qty.Speed.kt(0)

        return lps_dyn.State(
            position = lps_dyn.Displacement(lps_qty.Distance.m(0),
                                            lps_qty.Distance.m(0)),
            velocity = lps_dyn.Velocity(sonar_speed,
                                        lps_qty.Speed.kt(0)),
            acceleration = lps_dyn.Acceleration(lps_qty.Acceleration.m_s2(0),
                                                lps_qty.Acceleration.m_s2(0))
        )
