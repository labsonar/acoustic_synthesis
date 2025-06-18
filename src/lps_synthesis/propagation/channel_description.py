"""
This module provides classes for modeling and simulating acoustic wave propagation in underwater
environments.

Classes:
    Description: Handles the layers of an acoustic channel and their properties.
"""
import typing
import random
import json

import matplotlib.pyplot as plt
import numpy as np

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as lps_layer

class Description():
    """
    Represents the structure of an acoustic channel, including different layers
    (e.g., air, water, seabed) and the corresponding depths. The layers can be converted to a format
    compatible with the OASES software for further processing.
    """

    def __init__(self) -> None:
        self.air_sea = lps_layer.Air()
        self.layers = {}

    def to_oases_format(self) -> str:
        """
        Converts the acoustic channel description to OASES format, used for acoustic simulations.

        Returns:
            A string formatted according to the OASES input file format.

        Raises:
            UnboundLocalError: If the channel contains no layers.
        """
        if len(self.layers) == 0:
            raise UnboundLocalError("Should not export an empty channel")

        n_layers = len(self.layers) + (1 if self.air_sea is not None else 0)
        ret = f"{n_layers}\n"
        for depth, layer in self:
            ret += f"{depth.get_m():6f} {layer.to_oases_format()}\n"
        return ret[:-1]

    def __str__(self) -> str:
        ret = ""
        for depth, layer in self:
            ret += f"At {depth.get_m()}: {layer}\n"
        return ret[:-1]

    def __iter__(self) -> \
        typing.Iterator[typing.Tuple[lps_qty.Distance, lps_layer.AcousticalLayer]]:
        all_layers = [(lps_qty.Distance.m(0), self.air_sea)] if self.air_sea is not None else []
        all_layers += sorted(self.layers.items())
        return iter(all_layers)

    def add(self, depth: lps_qty.Distance,
            layer: typing.Union[lps_qty.Speed, lps_layer.AcousticalLayer]) -> None:
        """
        Adds a new layer to the channel description at a specific depth.

        Args:
            depth: The depth at which the new layer should be added.
            layer: The acoustical layer or sound speed to add at the given depth.
        """

        if isinstance(layer, lps_qty.Speed):
            self.add(depth=depth, layer=lps_layer.Water(sound_speed=layer))
        elif isinstance(layer, lps_layer.AcousticalLayer):
            self.layers[depth] = layer
        else:
            raise ValueError(("For add in AcousticalChannel, use lps_qty.Speed"
                             " or lps_layer.AcousticalLayer"))

    def remove(self, depth: lps_qty.Distance) -> None:
        """ Remove a layer to the channel description at a specific depth. """
        if depth in self.layers:
            self.layers.pop(depth)

    def remove_air_sea_interface(self) -> None:
        """ Remove the layer of air at 0 depth. """
        self.air_sea = None

    def export_ssp(self, filename: str) -> None:
        """
        Exports a sound speed profile plot (speed vs depth) based on the water layers.

        Args:
            filename: The file path where the plot should be saved.
        """
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

    def get_base_speed(self) -> lps_qty.Speed:
        """
        Retrieves the compressional speed of the first water layer (skipping air).

        Returns:
            The compressional speed of the first water layer, or a default of 1500 m/s if no water
            layer is present.
        """
        skip_air = True
        for _, layer in self:
            if skip_air:
                continue
            return layer.get_compressional_speed()

        return lps_qty.Speed.m_s(1500)

    def get_speed_at(self, desired_depth: lps_qty.Distance) -> lps_qty.Speed:
        """ Retrieves the sound speed at the specified depth. """

        last_speed = None
        last_depths = None

        for depth, layer in self:
            if not isinstance(layer, lps_layer.Water) :
                continue

            if depth > desired_depth:
                factor = (desired_depth - last_depths)/(depth - last_depths)
                return (1 - factor) * last_speed + factor * layer.get_compressional_speed()

            last_speed = layer.get_compressional_speed()
            last_depths = depth

        return last_speed

    @classmethod
    def get_random(cls) -> 'Description':
        """
        Creates a random acoustic channel description, with randomly generated depths,
        bottom types, and sound speed profiles. The number of layers and their properties
        are generated in a realistic way.

        Returns:
            A Description object with randomly generated layers.
        """
        desc = cls()

        max_depth = lps_qty.Distance.m(random.uniform(25, 1500))

        current_depth = lps_qty.Distance.m(0)
        speed = lps_qty.Speed.m_s(random.uniform(1480, 1530))
        desc.add(current_depth, speed)

        profile_type = random.choice(['positive', 'negative', 'iso'])

        mixing_layer_depth = int(random.uniform(25, 75))
        termocline_depth = int(random.uniform(mixing_layer_depth + 50,
                                              np.min([max_depth.get_m(), 1000])))

        mixing_n_layers = random.uniform(3, 10)
        termocline_n_layers = random.uniform(3, 10)
        depth_n_layers = random.uniform(3, 10)

        mixing_alpha = lps_qty.Speed.m_s(random.uniform(0.1, 0.3))
        termocline_alpha = lps_qty.Speed.m_s(random.uniform(0.1, 0.3))
        depth_alpha = lps_qty.Speed.m_s(random.uniform(0.1, 0.3))

        for depth in range(0, mixing_layer_depth, int(mixing_layer_depth/mixing_n_layers)):
            if depth >= max_depth.get_m():
                continue

            if profile_type == 'positive':
                speed = speed + mixing_alpha

            elif profile_type == 'negative':
                speed = speed - mixing_alpha

            desc.add(lps_qty.Distance.m(depth), speed)

        for depth in range(mixing_layer_depth, termocline_depth,
                           int((termocline_depth-mixing_layer_depth)/termocline_n_layers)):
            if depth >= max_depth.get_m():
                continue

            speed = speed - termocline_alpha

            desc.add(lps_qty.Distance.m(depth), speed)

        n_steps = int((max_depth.get_m()-termocline_depth)/depth_n_layers)
        if n_steps > 0:
            for depth in range(termocline_depth, int(max_depth.get_m()), n_steps):
                if depth >= max_depth.get_m():
                    continue

                speed = speed + depth_alpha
                desc.add(lps_qty.Distance.m(depth), speed)


            bottom_layer = random.choice([b for b in lps_layer.BottomType])
            desc.add(max_depth, bottom_layer)

        return desc

    @classmethod
    def get_default(cls) -> 'Description':
        """ Return a default Channel description. """
        desc = cls()
        desc.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))
        desc.add(lps_qty.Distance.m(50), lps_layer.BottomType.CHALK)
        return desc

    def save(self, file_path: str) -> None:
        """Saves the description to a JSON file."""
        local_dict = {}

        for depth, layer in self.layers.items():
            local_dict[depth.get_m()] = {
                'type': layer.__class__.__name__,
                'speed': layer.get_compressional_speed().get_m_s(),
                'value': layer.value if isinstance(layer, lps_layer.BottomType) else "None"
            }

        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(local_dict, file, indent=4)

    @staticmethod
    def load(file_path: str) -> 'Description':
        """ Load Channel description from a file. """

        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)

        desc = Description()

        for depth, layer in data.items():
            d = lps_qty.Distance.m(float(depth))

            if layer['type'] == lps_layer.BottomType.CLAY.__class__.__name__:
                desc.add(d, lps_layer.BottomType(int(layer['value'])))

            elif layer['type'] == lps_layer.Water().__class__.__name__:
                desc.add(d, lps_layer.Water(lps_qty.Speed.m_s(float(layer['speed']))))

            else:
                raise NotImplementedError(f"load for {layer['type']} not implemented")

        return desc
