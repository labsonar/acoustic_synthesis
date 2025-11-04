import enum
import random

import lps_utils.quantities as lps_qty

class DynamicType(enum.Enum):
    """Defines the motion type of the simulated event."""
    FIXED_DISTANCE = enum.auto()
    NEAR_CPA = enum.auto()
    FAR_CPA = enum.auto()

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
