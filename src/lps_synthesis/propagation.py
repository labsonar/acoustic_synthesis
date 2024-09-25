import enum
import typing
import random

import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty


class SeabedType(enum.Enum):
    #Table 1.3 - Computational Ocean Acoustics, Jensen
    CLAY = 1
    SILT = 2
    SAND = 3
    GRAVEL = 4
    MORAINE = 5
    CHALK = 6
    LIMESTONE = 7
    BASALT = 8

    def __str__(self) -> str:
        return self.name.lower()

    def get_compressional_speed(self) -> lps_qty.Speed:
        speed = { # cp/cw
            SeabedType.CLAY: 1,
            SeabedType.SILT: 1.05,
            SeabedType.SAND: 1.1,
            SeabedType.GRAVEL: 1.2,
            SeabedType.MORAINE: 1.3,
            SeabedType.CHALK: 1.6,
            SeabedType.LIMESTONE: 2,
            SeabedType.BASALT: 3.5,
        }
        return lps_qty.Speed.m_s(1500) * speed[self]

    def get_shear_speed(self) -> lps_qty.Speed:
        speed = { # cs (m/s)
            SeabedType.CLAY: 50,
            SeabedType.SILT: 80,
            SeabedType.SAND: 110,
            SeabedType.GRAVEL: 180,
            SeabedType.MORAINE: 600,
            SeabedType.CHALK: 1000,
            SeabedType.LIMESTONE: 1500,
            SeabedType.BASALT: 2500,
        }
        return lps_qty.Speed.m_s(speed[self])

    def get_compressional_attenuation(self) -> float:
        att = { # αp (dB/λ)
            SeabedType.CLAY: 0.2,
            SeabedType.SILT: 1.0,
            SeabedType.SAND: 0.8,
            SeabedType.GRAVEL: 0.6,
            SeabedType.MORAINE: 0.4,
            SeabedType.CHALK: 0.2,
            SeabedType.LIMESTONE: 0.1,
            SeabedType.BASALT: 0.1,
        }
        return att[self]

    def get_shear_attenuation(self) -> float:
        att = { # αs (dB/λ)
            SeabedType.CLAY: 1.0,
            SeabedType.SILT: 1.5,
            SeabedType.SAND: 2.5,
            SeabedType.GRAVEL: 1.5,
            SeabedType.MORAINE: 1.0,
            SeabedType.CHALK: 0.5,
            SeabedType.LIMESTONE: 0.2,
            SeabedType.BASALT: 0.2,
        }
        return att[self]

    def get_density(self) -> lps_qty.Density:
        density = { # ρb/pw
            SeabedType.CLAY: 1.5,
            SeabedType.SILT: 1.7,
            SeabedType.SAND: 1.9,
            SeabedType.GRAVEL: 2.0,
            SeabedType.MORAINE: 2.1,
            SeabedType.CHALK: 2.2,
            SeabedType.LIMESTONE: 2.4,
            SeabedType.BASALT: 2.7,
        }
        return lps_qty.Density.g_cm3(1) * density[self]

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

    def get_ordered_ssp(self) \
        -> typing.Tuple[typing.List[lps_qty.Distance], typing.List[lps_qty.Speed]]:

        paired_list = list(zip(self.depths, self.speeds))
        sort_idx = sorted(paired_list, key=lambda x: x[0].get_m())
        depths, speeds = zip(*sort_idx)
        depths = list(depths)
        speeds = list(speeds)
        return depths, speeds

    def get_max_depths(self) -> lps_qty.Distance:
        depths, _ = self.get_ordered_ssp()
        return depths[-1]

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
                 bottom: SeabedType,
                 bottom_depth: lps_qty.Distance = None) -> None:
        self.ssp = ssp
        self.bottom = bottom
        self.interpolation: float = random.random()
        self.bottom_depth = bottom_depth if bottom_depth is not None else ssp.get_max_depths()


#alternative alias
SSP = SoundSpeedProfile
BottomType = SeabedType
