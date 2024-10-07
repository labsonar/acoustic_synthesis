"""
This module provides a framework for modeling acoustical properties of layers in underwater
environments.

Classes and enums:
    - AcousticalLayer: Abstract base class for representing acoustical layers.
    - Water: Represents water as an acoustical layer.
    - Air: Represents air as an acoustical layer.
    - SeabedType: Enum for various seabed types, with acoustical properties.
    
The module uses `lps_utils.quantities` to represent physical quantities.
"""
import enum
import abc
import overrides

import lps_utils.quantities as lps_qty

class AcousticalLayer():
    """ Abstract class to represent an acoustical layer for channel modeling."""

    def to_oases_format(self) -> str:
        """ Layer in oasp description format. """
        return (f"{self.get_compressional_speed().get_m_s():6f} "
            f"{self.get_shear_speed().get_m_s():6f} "
            f"{self.get_compressional_attenuation():6f} "
            f"{self.get_shear_attenuation():6f} "
            f"{self.get_density().get_g_cm3():6f} "
            f"{self.get_rms_roughness().get_m():6f}")

    def __str__(self) -> str:
        return f"{type(self).__name__} layer with speed {str(self.get_compressional_speed())}"

    @abc.abstractmethod
    def get_compressional_speed(self) -> lps_qty.Speed:
        """Returns the compressional wave speed as a `lps_qty.Speed` object."""

    @abc.abstractmethod
    def get_shear_speed(self) -> lps_qty.Speed:
        """Returns the shear wave speed as a `lps_qty.Speed` object."""

    @abc.abstractmethod
    def get_compressional_attenuation(self) -> float:
        """Returns the compressional attenuation coefficient in dB/λ."""

    @abc.abstractmethod
    def get_shear_attenuation(self) -> float:
        """Returns the shear attenuation coefficient in dB/λ."""

    @abc.abstractmethod
    def get_density(self) -> lps_qty.Density:
        """Returns the density as a `lps_qty.Density` object."""

    @abc.abstractmethod
    def get_rms_roughness(self) -> lps_qty.Distance:
        """Returns the RMS roughness as a `lps_qty.Distance` object."""

class Water(AcousticalLayer):
    """ Represents water as an acoustical layer. """

    def __init__(self, sound_speed = lps_qty.Speed.m_s(1500)) -> None:
        super().__init__()
        self.sound_speed = sound_speed

    @overrides.overrides
    def get_compressional_speed(self) -> lps_qty.Speed:
        """Returns the compressional wave speed in water (default: 1500 m/s)."""
        return self.sound_speed

    @overrides.overrides
    def get_shear_speed(self) -> lps_qty.Speed:
        """Returns shear speed in water, which is 0 (no shear waves)."""
        return lps_qty.Speed.m_s(0)

    @overrides.overrides
    def get_compressional_attenuation(self) -> float:
        """Returns compressional attenuation in water, which is negligible (0 dB/λ)."""
        return 0

    @overrides.overrides
    def get_shear_attenuation(self) -> float:
        """Returns shear attenuation in water, which is 0 (no shear waves)."""
        return 0

    @overrides.overrides
    def get_density(self) -> lps_qty.Density:
        """Returns the density of water (1 g/cm³)."""
        return lps_qty.Density.g_cm3(1)

    @overrides.overrides
    def get_rms_roughness(self) -> lps_qty.Distance:
        """Returns the RMS roughness of the water surface, assumed to be 0 m."""
        return lps_qty.Distance.m(0)

class Air(AcousticalLayer):
    """ Represents air as an acoustical layer. """

    def __init__(self, sound_speed = lps_qty.Speed.m_s(340)) -> None:
        super().__init__()
        self.sound_speed = sound_speed

    @overrides.overrides
    def get_compressional_speed(self) -> lps_qty.Speed:
        """Returns the compressional wave speed in air (default: 340 m/s)."""
        return self.sound_speed

    @overrides.overrides
    def get_shear_speed(self) -> lps_qty.Speed:
        """Returns shear speed in air, which is 0 (no shear waves)."""
        return lps_qty.Speed.m_s(0)

    @overrides.overrides
    def get_compressional_attenuation(self) -> float:
        """Returns compressional attenuation in air, which is negligible (0 dB/λ)."""
        return 0

    @overrides.overrides
    def get_shear_attenuation(self) -> float:
        """Returns shear attenuation in air, which is 0 (no shear waves)."""
        return 0

    @overrides.overrides
    def get_density(self) -> lps_qty.Density:
        """Returns the density of air (1.225 kg/m³)."""
        return lps_qty.Density.kg_m3(1.225)

    @overrides.overrides
    def get_rms_roughness(self) -> lps_qty.Distance:
        """Returns the RMS roughness of the air surface, assumed to be 0 m."""
        return lps_qty.Distance.m(0)

class SeabedType(AcousticalLayer, enum.Enum):
    """
    Enum representing various types of seabed, each with distinct acoustical properties.

    Based on Table 1.3 - Computational Ocean Acoustics, Jensen
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
        return f"{self.name} layer with speed {str(self.get_compressional_speed())}"

    @overrides.overrides
    def get_compressional_speed(self) -> lps_qty.Speed:
        """
        Returns the compressional speed in seabed type as a `lps_qty.Speed`.
        """
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
        return lps_qty.Speed.m_s(1500 * speed_ratios[self])

    @overrides.overrides
    def get_shear_speed(self) -> lps_qty.Speed:
        """Returns the shear wave speed in seabed type as a `lps_qty.Speed`."""
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
        return lps_qty.Speed.m_s(shear_speeds[self])

    @overrides.overrides
    def get_compressional_attenuation(self) -> float:
        """Returns the compressional attenuation coefficient (dB/λ) for the seabed type."""
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
        return compressional_attenuations[self]

    @overrides.overrides
    def get_shear_attenuation(self) -> float:
        """Returns the shear attenuation coefficient (dB/λ) for the seabed type."""
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
        return shear_attenuations[self]

    @overrides.overrides
    def get_density(self) -> lps_qty.Density:
        """
        Returns the density of the seabed type as a `lps_qty.Density`.
        """
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
        return lps_qty.Density.g_cm3(1) * density_ratios[self]

    @overrides.overrides
    def get_rms_roughness(self) -> lps_qty.Distance:
        """Returns the RMS roughness of the seabed surface, assumed to be 0 m."""
        return lps_qty.Distance.m(0)

# Aliasing SeabedType to BottomType for easier reference.
BottomType = SeabedType
