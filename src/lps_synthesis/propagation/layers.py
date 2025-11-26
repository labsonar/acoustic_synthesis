"""
This module provides a framework for modeling acoustical properties of layers in underwater
environments.


The module uses `lps_utils.quantities` to represent physical quantities.
"""
import enum

import lps_utils.quantities as lps_qty
import lps_synthesis.environment.environment as lps_environment

class AcousticalLayer():
    """Generic class to represent an acoustical layer for channel modeling."""

    def __init__(self,
                 compressional_speed: lps_qty.Speed = None,
                 shear_speed: lps_qty.Speed = None,
                 compressional_attenuation: float = 0,
                 shear_attenuation: float = 0,
                 density: lps_qty.Density = None,
                 rms_roughness: lps_qty.Distance = None) -> None:
        self.compressional_speed = compressional_speed or lps_qty.Speed.m_s(0)
        self.shear_speed = shear_speed or lps_qty.Speed.m_s(0)
        self.compressional_attenuation = compressional_attenuation
        self.shear_attenuation = shear_attenuation
        self.density = density or lps_qty.Density.g_cm3(0)
        self.rms_roughness = rms_roughness or lps_qty.Distance.m(0)
        self._name = type(self).__name__

    def to_oases_format(self) -> str:
        """ Layer in oasp description format. """
        return (
            f"{self.get_compressional_speed().get_m_s():6f} "
            f"{self.get_shear_speed().get_m_s():6f} "
            f"{self.get_compressional_attenuation():6f} "
            f"{self.get_shear_attenuation():6f} "
            f"{self.get_density().get_g_cm3():6f} "
            f"{self.get_rms_roughness().get_m():6f}"
        )

    def __str__(self) -> str:
        return (
            f"{self._name} layer – cₚ={self.get_compressional_speed()} "
            f"/ ρ={self.get_density()}"
        )

    def get_compressional_speed(self) -> lps_qty.Speed:
        """Returns the compressional wave speed as a `lps_qty.Speed` object."""
        return self.compressional_speed

    def get_shear_speed(self) -> lps_qty.Speed:
        """Returns the shear wave speed as a `lps_qty.Speed` object."""
        return self.shear_speed

    def get_compressional_attenuation(self) -> float:
        """Returns the compressional attenuation coefficient in dB/λ."""
        return self.compressional_attenuation

    def get_shear_attenuation(self) -> float:
        """Returns the shear attenuation coefficient in dB/λ."""
        return self.shear_attenuation

    def get_density(self) -> lps_qty.Density:
        """Returns the density as a `lps_qty.Density` object."""
        return self.density

    def get_rms_roughness(self) -> lps_qty.Distance:
        """Returns the RMS roughness as a `lps_qty.Distance` object."""
        return self.rms_roughness


class Water(AcousticalLayer):
    """ Represents water as an acoustical layer. """
    def __init__(self,
                 sound_speed: lps_qty.Speed = None,
                 density: lps_qty.Density = None) -> None:
        sound_speed = sound_speed or lps_qty.Speed.m_s(1500)
        density = density or lps_qty.Density.g_cm3(1)
        super().__init__(
            compressional_speed=sound_speed,
            shear_speed=lps_qty.Speed.m_s(0),
            compressional_attenuation=0,
            shear_attenuation=0,
            density=density,
            rms_roughness=lps_qty.Distance.m(0))


class Air(AcousticalLayer):
    """ Represents air as an acoustical layer. """

    def __init__(self, sea_state: lps_environment.Sea = None)-> None:
        sea_state = sea_state or lps_environment.Sea.STATE_0
        super().__init__(
            compressional_speed=lps_qty.Speed.m_s(340),
            shear_speed=lps_qty.Speed.m_s(0),
            compressional_attenuation=0,
            shear_attenuation=0,
            density=lps_qty.Density.kg_m3(1.225),
            rms_roughness=sea_state.get_rms_roughness())


class SeabedType(enum.Enum):
    """
    Enum representing various types of seabed, each with distinct acoustical properties.
    """
    CLAY = 1
    SILT = 2
    SAND = 3
    GRAVEL = 4
    MORAINE = 5
    CHALK = 6
    LIMESTONE = 7
    BASALT = 8

    def __str__(self) -> str:
        return self.name.title()

    def get_acoustical_layer(self) -> 'Seabed':
        """ Get a valid acoustical layer for this SeabedType. """
        return Seabed(self)


class Seabed(AcousticalLayer):
    """
    Class to represent a acoustical layer of a seabed type.

    Based on Table 1.3 - Computational Ocean Acoustics, Jensen
    """
    def __init__(self, seabed_type: SeabedType, seed: int = None):
        self.seed = seed if seed is not None else id(self)

        self.seabed_type = seabed_type
        super().__init__(
            compressional_speed = self._sort_compressional_speed(),
            shear_speed = self._sort_shear_speeds(),
            compressional_attenuation = self._sort_compressional_attenuations(),
            shear_attenuation = self._sort_shear_attenuations(),
            density = self._sort_density(),
            rms_roughness = self._draw_rms_roughness(),
        )

    def __str__(self) -> str:
        return str(self.seabed_type)

    def to_complete_str(self) -> str:
        """ More complete print """
        return f"{self.seabed_type} layer with speed {str(self.get_compressional_speed())}"

    def _sort_compressional_speed(self) -> lps_qty.Speed:
        speed_ratios = {
            SeabedType.CLAY: 1,
            SeabedType.SILT: 1.05,
            SeabedType.SAND: 1.1,
            SeabedType.GRAVEL: 1.2,
            SeabedType.MORAINE: 1.3,
            SeabedType.CHALK: 1.6,
            SeabedType.LIMESTONE: 2,
            SeabedType.BASALT: 3.5,
        }
        return lps_qty.Speed.m_s(1500 * speed_ratios[self.seabed_type])

    def _sort_shear_speeds(self) -> lps_qty.Speed:
        shear_speeds = {
            SeabedType.CLAY: 50,
            SeabedType.SILT: 80,
            SeabedType.SAND: 110,
            SeabedType.GRAVEL: 180,
            SeabedType.MORAINE: 600,
            SeabedType.CHALK: 1000,
            SeabedType.LIMESTONE: 1500,
            SeabedType.BASALT: 2500,
        }
        return lps_qty.Speed.m_s(shear_speeds[self.seabed_type])

    def _sort_compressional_attenuations(self) -> float:
        compressional_attenuations = {
            SeabedType.CLAY: 0.2,
            SeabedType.SILT: 1.0,
            SeabedType.SAND: 0.8,
            SeabedType.GRAVEL: 0.6,
            SeabedType.MORAINE: 0.4,
            SeabedType.CHALK: 0.2,
            SeabedType.LIMESTONE: 0.1,
            SeabedType.BASALT: 0.1,
        }
        return compressional_attenuations[self.seabed_type]

    def _sort_shear_attenuations(self) -> float:
        shear_attenuations = {
            SeabedType.CLAY: 1.0,
            SeabedType.SILT: 1.5,
            SeabedType.SAND: 2.5,
            SeabedType.GRAVEL: 1.5,
            SeabedType.MORAINE: 1.0,
            SeabedType.CHALK: 0.5,
            SeabedType.LIMESTONE: 0.2,
            SeabedType.BASALT: 0.2,
        }
        return shear_attenuations[self.seabed_type]

    def _sort_density(self) -> lps_qty.Density:
        density_ratios = {
            SeabedType.CLAY: 1.5,
            SeabedType.SILT: 1.7,
            SeabedType.SAND: 1.9,
            SeabedType.GRAVEL: 2.0,
            SeabedType.MORAINE: 2.1,
            SeabedType.CHALK: 2.2,
            SeabedType.LIMESTONE: 2.4,
            SeabedType.BASALT: 2.7,
        }
        return lps_qty.Density.g_cm3(1) * density_ratios[self.seabed_type]

    def _draw_rms_roughness(self) -> lps_qty.Distance:
        """Returns a valid RMS roughness value for the seabed surface."""
        # roughness_range = {
        #     SeabedType.CLAY: (0, 9.75e-6),
        #     SeabedType.SILT: (9.75e-6, 1.5625e-3),
        #     SeabedType.SAND: (1.5625e-4, 5e-3),
        #     SeabedType.GRAVEL: (5e-3, 625e-3),
        #     SeabedType.MORAINE: (0.5, 4),
        #     SeabedType.CHALK: (0, 2.5e-6),
        #     SeabedType.LIMESTONE: (0.396, 0.492),
        #     SeabedType.BASALT: (99, 259),
        # }
        # rng = random.Random(self.seed)
        # value = rng.uniform(*(roughness_range[self.seabed_type]))
        # return lps_qty.Distance.m(value)
        return lps_qty.Distance.m(0)

# Aliasing SeabedType to BottomType for easier reference.
BottomType = SeabedType
