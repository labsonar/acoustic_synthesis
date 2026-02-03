"""
Dynamic Module

Define the dynamics and their implementations for the synthetic dataset.
"""
import enum
import typing
import random

import pandas as pd

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.noise_source as lps_noise
import lps_synthesis.scenario.dynamic as lps_dyn
import lps_synthesis.database.catalog as syndb_core

class DynamicType(enum.Enum):
    """Defines the motion type of the simulated event."""
    FIXED_DISTANCE = enum.auto()
    CPA_IN = enum.auto()
    CPA_OUT = enum.auto()

    def __str__(self):
        return self.name.lower()

class SimulationDynamic(syndb_core.CatalogEntry):
    """ Describes the dynamic behavior of the acoustic simulation. """

    def __init__(self, dynamic_type: DynamicType, shortest: lps_qty.Distance, approaching: bool):
        self.dynamic_type = dynamic_type
        self.shortest = shortest
        self.approaching = approaching

    def __str__(self):
        return f"{self.dynamic_type.name} (d_min={self.shortest})"

    def as_dict(self) -> dict[str, typing.Any]:
        """ Converts the entry into a dictionary suitable for tabular representation. """
        return {
            "DYNAMIC_TYPE": str(self.dynamic_type),
            "SHORTEST_DIST_M": self.shortest.get_m(),
            "APPROACHING": self.approaching
        }

    @classmethod
    def load_catalog(cls, filename: str) -> syndb_core.Catalog["SimulationDynamic"]:
        df = pd.read_csv(filename)
        dynamics = []

        for _, row in df.iterrows():
            dynamic_type = DynamicType[row["DYNAMIC_TYPE"].upper()]
            shortest = lps_qty.Distance.m(row["SHORTEST_DIST_M"])
            approaching = bool(row["APPROACHING"])

            dynamics.append(
                SimulationDynamic(
                    dynamic_type = dynamic_type,
                    shortest = shortest,
                    approaching = approaching,
                )
            )

        return syndb_core.Catalog[SimulationDynamic](entries=dynamics)

    @staticmethod
    def rand_catalog(n_samples: int,
                    min_dist: lps_qty.Distance = lps_qty.Distance.m(50),
                    max_dist: lps_qty.Distance = lps_qty.Distance.m(250),
                    seed: int = 42
                    ) -> syndb_core.Catalog["SimulationDynamic"]:
        """Generate a catalog with random dynamic configurations."""

        rng = random.Random(seed)
        dynamics: list[SimulationDynamic] = []

        for _ in range(n_samples):

            dist = lps_qty.Distance.m(random.randint(int(min_dist.get_m()),
                                                    int(max_dist.get_m())))
            dynamic_type = rng.choice(list(DynamicType))
            approaching = rng.random() < 0.5

            dynamics.append(
                SimulationDynamic(dynamic_type, dist, approaching)
            )

        return syndb_core.Catalog[SimulationDynamic](entries=dynamics)


    def get_ship_initial_state(self,
                               speed: lps_qty.Speed,
                               step_interval: lps_qty.Time,
                               simulation_steps: int) -> lps_dyn.State:
        """ Define the initial state to fulfill the dynamics. """

        y_offset = self.shortest if self.dynamic_type != DynamicType.CPA_OUT \
                                 else lps_qty.Distance.m(0)

        displacement = speed * (step_interval * (simulation_steps-1))

        if self.dynamic_type == DynamicType.FIXED_DISTANCE:
            x_offset = lps_qty.Distance.m(0)
        elif self.dynamic_type == DynamicType.CPA_IN:
            x_offset = -0.5 * displacement
        elif self.dynamic_type == DynamicType.CPA_OUT:
            if self.approaching:
                x_offset = (self.shortest + displacement) * -1
            else:
                x_offset = self.shortest
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
