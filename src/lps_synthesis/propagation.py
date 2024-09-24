import enum
import typing
import random

import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty


class SeabedType(enum.Enum):
    SILT = 1
    SAND = 2
    GRAVEL = 3
    MUD = 4
    ROCK = 5
    CORAL = 6
    CALCAREOUS_OOZE = 7
    GLACIAL_TILL = 8


    def get_speed(self, interpolation: float = random.random()):
        min_speed = {
            SeabedType.SILT: 1450,
            SeabedType.SAND: 1600,
            SeabedType.GRAVEL: 1800,
            SeabedType.MUD: 1450,
            SeabedType.ROCK: 3000,
            SeabedType.CORAL: 1700,
            SeabedType.CALCAREOUS_OOZE: 1450,
            SeabedType.GLACIAL_TILL: 1700,
        }
        max_speed = {
            SeabedType.SILT: 1550,
            SeabedType.SAND: 1800,
            SeabedType.GRAVEL: 2000,
            SeabedType.MUD: 1550,
            SeabedType.ROCK: 5000,
            SeabedType.CORAL: 2300,
            SeabedType.CALCAREOUS_OOZE: 1550,
            SeabedType.GLACIAL_TILL: 2100,
        }
        return lps_qty.Speed.m_s(min_speed[self] * (1-interpolation) + max_speed[self] * interpolation)

    def get_density(self, interpolation: float = random.random()):
        min_density = {
            SeabedType.SILT: 1.3,
            SeabedType.SAND: 1.8,
            SeabedType.GRAVEL: 2.0,
            SeabedType.MUD: 1.5,
            SeabedType.ROCK: 2.5,
            SeabedType.CORAL: 2.0,
            SeabedType.CALCAREOUS_OOZE: 1.4,
            SeabedType.GLACIAL_TILL: 2.1,
        }
        max_density = {
            SeabedType.SILT: 1.7,
            SeabedType.SAND: 2.1,
            SeabedType.GRAVEL: 2.3,
            SeabedType.MUD: 1.7,
            SeabedType.ROCK: 2.7,
            SeabedType.CORAL: 2.4,
            SeabedType.CALCAREOUS_OOZE: 1.6,
            SeabedType.GLACIAL_TILL: 2.4,
        }
        return lps_qty.Density.g_cm3(min_density[self] * (1-interpolation) + max_density[self] * interpolation)

class SoundSpeedProfile:
    def __init__(self,
                 depths: typing.List[lps_qty.Distance] = [],
                 speeds: typing.List[lps_qty.Speed] = []):
        self.depths = depths
        self.speeds = speeds

        if len(depths) != len(speeds):
            raise UnboundLocalError("SoundSpeedProfile must be made by pairs of depths and speed")

    def add(self, depth: lps_qty.Distance, speed: lps_qty.Speed):
        self.depths.append(depth)
        self.speeds.append(speed)

    def __str__(self) -> str:
        ret = ""
        for depth, speed in zip(self.depths, self.speeds):
            ret = f"{ret}[{depth}]: {speed}\n"
        return ret[:-1]

    def print(self, filename: str) -> None:

        depths = []
        speeds = []
        for depth, speed in zip(self.depths, self.speeds):
            depths.append(depth.get_m())
            speeds.append(speed.get_m_s())

        plt.plot(speeds, depths)
        plt.gca().invert_yaxis()
        plt.savefig(filename)
        plt.close()

class AcousticalChannel():
    def __init__(self,
                 ssp: SoundSpeedProfile,
                 bottom: SeabedType) -> None:
        self.ssp = ssp
        self.bottom = bottom
        self.interpolation: float = random.random()




#alternative alias
SSP = SoundSpeedProfile
BottomType = SeabedType