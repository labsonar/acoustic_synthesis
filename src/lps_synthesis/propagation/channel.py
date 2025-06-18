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
import enum

import numpy as np

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.channel_description as lps_desc
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.models as lps_model

# DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".lps", "channel")
DEFAULT_DIR = os.path.join(".", "channel")

class Channel():
    """
    Represents an acoustic channel that computes transfer functions and handles data storage.
    """

    def __init__(self,
                 description: lps_desc.Description,
                 sensor_depth: lps_qty.Distance,
                 source_depths: typing.List[lps_qty.Distance] =
                                [lps_qty.Distance.m(d) for d in np.arange(5, 25, 2)],
                 max_distance: lps_qty.Distance = lps_qty.Distance.km(1),
                 max_distance_points: int = 128,
                 sample_frequency: lps_qty.Frequency = lps_qty.Frequency.khz(16),
                 frequency_range: typing.Tuple[lps_qty.Frequency] = None,
                 model: lps_model.Model = lps_model.Model.OASES,
                 channel_dir: typing.Optional[str] = None,
                 hash_id: str = None):

        self.description = description
        self.source_depths = source_depths
        self.sensor_depth = sensor_depth
        self.max_distance = max_distance
        self.max_distance_points = max_distance_points
        self.sample_frequency = sample_frequency
        self.frequency_range = frequency_range
        self.model = model
        self.channel_dir = channel_dir if channel_dir is not None else DEFAULT_DIR
        self.hash_id = hash_id

        os.makedirs(self.channel_dir, exist_ok=True)

        self.response = None

        if not self._load():
            self._calc()

    def _filename(self, ext: str  = ".pkl") -> str:
        return os.path.join(self.channel_dir, f"{self._get_hash()}{ext}")

    def _load(self) -> bool:
        self.response = lps_model.ImpulseResponse.load(self._filename())
        return self.response is not None

    def _calc(self) -> None:
        self.response = self.model.estimate_transfer_function(
                        description = self.description,
                        source_depth = self.source_depths,
                        sensor_depth = self.sensor_depth,
                        max_distance = self.max_distance,
                        max_distance_points = self.max_distance_points,
                        sample_frequency = self.sample_frequency,
                        frequency_range = self.frequency_range,
                        filename = self._filename(ext=".dat"))
        self.response.save(self._filename())

    def _get_hash(self) -> str:
        if self.hash_id is not None:
            return self.hash_id

        hash_dict = {
            'description': self.description.to_oases_format(),
            'source_depths': [d.get_m() for d in self.source_depths],
            'sensor_depth': self.sensor_depth.get_m(),
            'max_distance': self.max_distance.get_m(),
            'distance_points': self.max_distance_points,
            'sample_frequency': self.sample_frequency.get_hz(),
            'model': self.model.name,
            'frequency_range': "None" if self.frequency_range is None else \
                                    [f.get_hz() for f in self.frequency_range]
        }
        converted = json.dumps(hash_dict, sort_keys=True)
        hash_obj = hashlib.md5(converted.encode())
        return hash_obj.hexdigest()

    def propagate(self,
                  input_data: np.array,
                  source_depth: lps_qty.Distance,
                  distance: typing.List[lps_qty.Distance]) -> np.array:
        """ Calculates the signal after propagation in the channel

        Args:
            input_data (np.array): Signal to propagate, should have size (n_samples,)
            source_depth (lps_qty.Distance): Depth of the source (ship's draft)
            distance (typing.List[lps_qty.Distance]): Equivalent distance for input_data.
                If the number of distances is different from the number of samples in the input
                data, the distances are interpolated.

        Returns:
            np.array: signal after propagation in the channel
        """
        return self.response.apply(input_data = input_data,
                                   source_depth = source_depth,
                                   distance = distance)

    def get_ir(self) -> lps_model.ImpulseResponse:
        """ Return the impulse response of a channel. """
        return self.response


class PredefinedChannel(enum.Enum):
    """ Enum class to represent predefined and preestimated channels. """
    BASIC = 0
    DUMMY = 1
    DEEPSHIP = 2
    DELTA_NODE = 2

    def get_channel(self):
        """ Return the estimated channel"""

        if self == PredefinedChannel.BASIC:

            desc = lps_desc.Description()
            desc.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))
            desc.add(lps_qty.Distance.m(50), lps_layer.BottomType.CHALK)

            return Channel(description = desc,
                            sensor_depth = lps_qty.Distance.m(40),
                            max_distance = lps_qty.Distance.km(1),
                            max_distance_points = 128,
                            sample_frequency = lps_qty.Frequency.khz(16),
                            frequency_range = None,
                            model = lps_model.Model.OASES,
                            hash_id=self.name.lower())

        if self == PredefinedChannel.DUMMY:

            desc = lps_desc.Description()
            desc.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))
            desc.add(lps_qty.Distance.m(1000), lps_qty.Speed.m_s(1500))
            desc.remove_air_sea_interface()

            return Channel(description = desc,
                            sensor_depth = lps_qty.Distance.m(500),
                            source_depths = [lps_qty.Distance.m(500)],
                            max_distance = lps_qty.Distance.m(1000),
                            max_distance_points = 110,
                            sample_frequency = lps_qty.Frequency.khz(16),
                            frequency_range = None,
                            model = lps_model.Model.OASES,
                            hash_id=self.name.lower())

        if self == PredefinedChannel.DELTA_NODE:

            desc = lps_desc.Description()

            desc.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))

            mixing_layer_depth = 30
            for depth in range(0, mixing_layer_depth, 10):
                desc.add(lps_qty.Distance.m(depth), lps_qty.Speed.m_s(1490 + depth * 0.3))

            termocline_depth = 80
            for depth in range(mixing_layer_depth, termocline_depth, 10):
                desc.add(lps_qty.Distance.m(depth), lps_qty.Speed.m_s(1500 - depth * 0.2))

            for depth in range(termocline_depth, 200, 10):
                desc.add(lps_qty.Distance.m(depth), lps_qty.Speed.m_s(1495 + depth * 0.1))

            desc.add(lps_qty.Distance.m(200), lps_layer.SeabedType.SILT)

            return Channel(description = desc,
                            sensor_depth = lps_qty.Distance.m(140),
                            max_distance = lps_qty.Distance.km(1),
                            max_distance_points = 128,
                            sample_frequency = lps_qty.Frequency.khz(16),
                            frequency_range = None,
                            model = lps_model.Model.OASES,
                            hash_id=self.name.lower())

        else:
            raise NotImplementedError(f"PredefinedChannel {self} not implemented")
