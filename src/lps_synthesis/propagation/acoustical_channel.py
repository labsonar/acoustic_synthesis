import enum
import typing
import abc, overrides
import random

import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as lps_layer

class Description():

    def __init__(self) -> None:
        self.air_sea = lps_layer.Air()
        self.layers = {}

    def to_oases_format(self) -> str:
        ret = ""
        for depth, layer in self:
            ret += f"{depth.get_m():6f} {layer.to_oases_format()}\n"
        return ret[:-1]

    def __str__(self) -> str:
        ret = ""
        for depth, layer in self:
            ret += f"At {depth.get_m()}: {layer}\n"
        return ret[:-1]

    def add(self, depth: lps_qty.Distance, layer: typing.Union[lps_qty.Speed, lps_layer.AcousticalLayer]):

        if isinstance(layer, lps_qty.Speed):
            self.add(depth=depth, layer=lps_layer.Water(sound_speed=layer))
        elif isinstance(layer, lps_layer.AcousticalLayer):
            self.layers[depth] = layer
        else:
            raise ValueError(("For add in AcousticalChannel, use lps_qty.Speed"
                             " or lps_layer.AcousticalLayer"))

    def __iter__(self):
        all_layers = [(lps_qty.Distance.m(0), self.air_sea)]
        all_layers += sorted(self.layers.items())        
        return iter(all_layers)

    def export_ssp(self, filename: str) -> None:
        depths = []
        speeds = []
        for depth, layer in self:
            if isinstance(layer, lps_layer.Water):
                depths.append(depth.get_m())
                speeds.append(layer.get_compressional_speed().get_m_s())

        plt.plot(speeds, depths)
        plt.gca().invert_yaxis()
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Depth (m)")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()