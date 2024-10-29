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
import bisect

import matplotlib.pyplot as plt
import numpy as np

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.channel_description as lps_desc
import lps_synthesis.propagation.models as lps_model

class Channel():
    """
    Represents an acoustic channel that computes transfer functions and handles data storage.
    """

    def __init__(self,
                 description: lps_desc.Description,
                 source_depths: typing.List[lps_qty.Distance],
                 sensor_depth: lps_qty.Distance,
                 max_distance: lps_qty.Distance = lps_qty.Distance.km(1),
                 max_distance_points: int = 128,
                 sample_frequency: lps_qty.Frequency = lps_qty.Frequency.khz(16),
                 frequency_range: typing.Tuple[lps_qty.Frequency] = None,
                 model: lps_model.Model = lps_model.Model.OASES,
                 temp_dir: str = "."):

        os.makedirs(temp_dir, exist_ok=True)

        self.description = description
        self.source_depth = source_depths
        self.sensor_depth = sensor_depth
        self.max_distance = max_distance
        self.max_distance_points = max_distance_points
        self.sample_frequency = sample_frequency
        self.frequency_range = frequency_range
        self.model = model
        self.temp_dir = temp_dir

        self.response = None

        if not self._load():
            self._calc()

    def _filename(self) -> str:
        return os.path.join(self.temp_dir, f"{self._get_hash()}.pkl")

    def _load(self) -> bool:
        self.response = lps_model.ImpulseResponse.load(self._filename())
        return self.response is not None

    def _calc(self) -> None:
        self.response = self.model.estimate_transfer_function(
                        description = self.description,
                        source_depth = self.source_depth,
                        sensor_depth = self.sensor_depth,
                        max_distance = self.max_distance,
                        max_distance_points = self.max_distance_points,
                        sample_frequency = self.sample_frequency,
                        frequency_range = self.frequency_range,
                        filename = os.path.join(self.temp_dir, "test.dat"))
        self.response.save(self._filename())

    def _get_hash(self) -> str:
        hash_dict = {
            'description': self.description.to_oases_format(),
            'source_depths': [d.get_m() for d in self.source_depth],
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

        depth_index = bisect.bisect_left(self.response.depths, source_depth)

        if depth_index == len(self.response.depths):
            depth_index -= 1
        elif depth_index != 0 and (self.response.depths[depth_index] - source_depth > \
                                   source_depth - self.response.depths[depth_index - 1]):
            depth_index -= 1

        h_t_tau = self.response.h_t_tau[depth_index].T[::-1]

        time_response = h_t_tau.shape[0]

        x = np.concatenate((np.zeros(time_response-1), input_data))
        y = np.zeros_like(input_data, dtype=np.complex_)

        dists = [np.abs(d.get_m()) for d in distance]
        if len(input_data) != len(dists):
            dists = np.interp(np.linspace(0, 1, len(input_data)),
                              np.linspace(0, 1, len(dists)),
                              dists)

        ranges = [r.get_m() for r in self.response.ranges]

        for y_i in range(len(input_data)):
            r_i = bisect.bisect_right(ranges, dists[y_i])
            interp_factor = (dists[y_i] - ranges[r_i-1])/(ranges[r_i] - ranges[r_i-1])

            ir = (1 - interp_factor) * h_t_tau[:, r_i - 1] + interp_factor * h_t_tau[:, r_i]
            y[y_i] = np.dot(x[y_i:y_i + time_response], ir)

        y = np.real(y)

        return y

    def export_h_f(self, filename: str, source_id: int = 0) -> None:
        """
        Exports the transfer function H(f) as an image, with distance and frequency axes.

        Args:
            filename: The file path where the image should be saved.
        """
        plt.imshow(abs(self.response.h_f_tau[source_id,:,:self.response.h_f_tau.shape[2]//2]),
                   aspect='auto', cmap='jet',
                   extent=[self.response.frequencies[0].get_khz(),
                        self.response.frequencies[len(self.response.frequencies)//2].get_khz(),
                        self.response.ranges[0].get_km(),
                        self.response.ranges[-1].get_km()]
        )

        plt.xlabel("Frequency (kHz)")
        plt.ylabel("Distance (km)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def export_h_t_tau(self, filename: str, source_id: int = 0) -> None:
        """
        Exports the transfer function h(t) as an image, with distance and time axes.

        Args:
            filename: The file path where the image should be saved.
        """

        plt.imshow(abs(self.response.h_t_tau[source_id,:,:]), aspect='auto',
                    cmap='jet', interpolation='none',
                    extent=[
                                    self.response.times[0].get_s(),
                                    self.response.times[-1].get_s(),
                                    self.response.ranges[-1].get_m(),
                                    self.response.ranges[0].get_m()]
                    )

        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")

        plt.colorbar()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def export_h_t_plots(self, filename: str, n_plots: int = 16, source_id: int = 0) -> None:
        """
        Exports the transfer function h(t) as an image, with intensity and time axes.

        Args:
            filename: The file path where the image should be saved.
            n_plots: number of plots equally separated in ranges.
        """

        times = [t.get_s() for t in self.response.times]
        labels = []
        for r_i in range(0, len(self.response.ranges), len(self.response.ranges)//n_plots):
            plt.plot(times, abs(self.response.h_t_tau[source_id,r_i,:]))
            labels.append(str(self.response.ranges[r_i]))

        plt.legend(labels)
        plt.savefig(filename)
        plt.close()
