"""
This module provides classes for modeling and simulating acoustic wave propagation in underwater
environments.

Classes:
    Description: Handles the layers of an acoustic channel and their properties.
    Channel: Manages the computation and storage of transfer functions using the channel description
"""
import os
import typing
import json
import hashlib
import pickle

import matplotlib.pyplot as plt

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

        ret = f"{len(self.layers) + 1}\n"
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
        all_layers = [(lps_qty.Distance.m(0), self.air_sea)]
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

class Channel():
    """
    Represents an acoustic channel that computes transfer functions and handles data storage.
    """

    def __init__(self,
                 description: Description,
                 source_depth: lps_qty.Distance,
                 sensor_depth: lps_qty.Distance,
                 max_distance: lps_qty.Distance = lps_qty.Distance.km(1),
                 distance_points: int = 128,
                 sample_frequency: lps_qty.Frequency = lps_qty.Frequency.khz(16),
                 n_fft: int = 128,
                 temp_dir: str = "."):

        os.makedirs(temp_dir, exist_ok=True)

        self.description = description
        self.source_depth = source_depth
        self.sensor_depth = sensor_depth
        self.max_distance = max_distance
        self.distance_points = distance_points
        self.sample_frequency = sample_frequency
        self.n_fft = n_fft
        self.temp_dir = temp_dir

        self.h_f = None
        self.h_t = None
        self.ranges = None
        self.times = None
        self.frequencies = None

        if not self._load():
            self._calc()

    def _filename(self) -> str:
        return os.path.join(self.temp_dir, f"{self._get_hash()}.pkl")

    def _load(self) -> bool:
        if os.path.exists(self._filename()):
            with open(self._filename(), 'rb') as file:
                self.h_f, self.h_t, self.ranges, self.times, self.frequencies = pickle.load(file)
                return True
        return False

    def _calc(self) -> None:

        from lps_synthesis.propagation.oases import estimate_transfer_function

        self.h_f, self.h_t, self.ranges, self.times, self.frequencies = estimate_transfer_function(
                description = self.description,
                source_depth = self.source_depth,
                sensor_depth = self.sensor_depth,
                max_distance = self.max_distance,
                distance_points = self.distance_points,
                sample_frequency = self.sample_frequency,
                n_fft = self.n_fft,
                filename = os.path.join(self.temp_dir, "test.dat"))

        with open(self._filename(), 'wb') as file:
            pickle.dump((self.h_f, self.h_t, self.ranges, self.times, self.frequencies), file)

    def _get_hash(self) -> str:
        hash_dict = {
            'description': self.description.to_oases_format(),
            'source_depth': self.source_depth.get_m(),
            'sensor_depth': self.sensor_depth.get_m(),
            'max_distance': self.max_distance.get_m(),
            'distance_points': self.distance_points,
            'sample_frequency': self.sample_frequency.get_hz(),
            'n_fft': self.n_fft
        }
        converted = json.dumps(hash_dict, sort_keys=True)
        hash_obj = hashlib.md5(converted.encode())
        return hash_obj.hexdigest()

    def export_h_f(self, filename: str) -> None:
        """
        Exports the transfer function H(f) as an image, with distance and frequency axes.

        Args:
            filename: The file path where the image should be saved.
        """
        plt.imshow(abs(self.h_f), extent=[self.ranges[0].get_m(),
                                self.ranges[-1].get_m(),
                                self.frequencies[0].get_hz(),
                                self.frequencies[-1].get_hz()], aspect='auto', cmap='jet')

        plt.xlabel("Distance (m)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def export_h_t_fig(self, filename: str) -> None:
        """
        Exports the transfer function h(t) as an image, with distance and time axes.

        Args:
            filename: The file path where the image should be saved.
        """

        plt.imshow(abs(self.h_t).T, extent=[
                                0,
                                len(self.times),
                                self.ranges[-1].get_m(),
                                self.ranges[0].get_m()], aspect='auto', cmap='jet')

        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def export_h_t_plots(self, filename: str, n_plots: int = 16) -> None:
        """
        Exports the transfer function h(t) as an image, with intensity and time axes.

        Args:
            filename: The file path where the image should be saved.
            n_plots: number of plots equally separated in ranges.
        """

        labels = []
        for r_i in range(0, len(self.ranges), n_plots):
            plt.plot(self.times, abs(self.h_t[:,r_i]))
            labels.append(str(self.ranges[r_i]))

        plt.legend(labels)
        plt.savefig(filename)
        plt.close()
