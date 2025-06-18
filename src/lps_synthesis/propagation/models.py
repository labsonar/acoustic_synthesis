"""Module for representing the channel impulse response and methods for estimating them
"""
import os
import enum
import typing
import pickle
import bisect

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scipy

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.channel_description as lps_channel
import lps_synthesis.propagation.oases as oases


def apply_doppler(input_data: np.array,
                  speeds: typing.List[lps_qty.Speed],
                  sound_speed: lps_qty.Speed) -> typing.Tuple[np.array, int]:
    """ Applies time-varying doppler based on approach speed.

    Args:
        input_data (np.array): Input data
        speeds (typing.List[lps_qty.Speed]): The speed across the data should be equivalent to a
            block slice of input_data.
        sound_speed (lps_qty.Speed): Reference sound speed propagation

    Returns:
        np.array: output data with zero padding to keep size and the number of efective samples
    """

    input_samples = len(input_data)
    num_blocks = len(speeds)
    samples_per_block = input_samples//num_blocks

    output = np.array([])
    for i_block in range(num_blocks):
        doppler_factor = (sound_speed + speeds[i_block]) / sound_speed

        scaled_data = scipy.resample(
                input_data[i_block * samples_per_block:(i_block+1) * samples_per_block - 1],
                int(samples_per_block//doppler_factor))

        output = np.concatenate((output, scaled_data))

    # output_samples = len(output)
    # output = np.pad(output, (0, output_samples-input_samples), 'constant')
    return output


class ImpulseResponse():
    """ Simple class to represent all the data needed to represent a response. """
    def __init__(self,
                 h_t_tau: np.array = None,
                 depths: typing.List[lps_qty.Distance] = None,
                 ranges: typing.List[lps_qty.Distance] = None,
                 frequencies: typing.List[lps_qty.Frequency] = None,
                 times: typing.List[lps_qty.Time] = None):
        self.h_t_tau = h_t_tau
        self.depths = depths
        self.ranges = ranges
        self.frequencies = frequencies
        self.times = times

    @classmethod
    def load(cls, filename: str) -> 'ImpulseResponse':
        """ Load a impulse response from file. """
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                ir = cls()
                ir.h_t_tau, ir.depths, ir.ranges, ir.times, ir.frequencies = pickle.load(file)
                return ir
        return None

    def save(self, filename: str) -> None:
        """ Save a impulse response to file. """
        with open(filename, 'wb') as file:
            pickle.dump((self.h_t_tau, self.depths, self.ranges, self.times, self.frequencies),
                        file)

    def apply(self,
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

        depth_index = bisect.bisect_left(self.depths, source_depth)

        if depth_index == len(self.depths):
            depth_index -= 1
        elif depth_index != 0 and (self.depths[depth_index] - source_depth > \
                                   source_depth - self.depths[depth_index - 1]):
            depth_index -= 1

        h_t_tau = self.h_t_tau[depth_index].T[::-1]

        n_samples = h_t_tau.shape[0]

        x = np.concatenate((np.zeros(n_samples-1), input_data))
        # y = np.zeros_like(input_data, dtype=np.complex_)
        y = np.zeros_like(input_data)

        dists = [np.abs(d.get_m()) for d in distance]
        if len(input_data) != len(dists):
            dists = np.interp(np.linspace(0, 1, len(input_data)),
                              np.linspace(0, 1, len(dists)),
                              dists)

        ranges = [r.get_m() for r in self.ranges]

        last_distance = None
        ir = None

        for y_i in range(len(input_data)):
            if last_distance != dists[y_i]:
                r_i = bisect.bisect_right(ranges, dists[y_i])
                interp_factor = (dists[y_i] - ranges[r_i-1])/(ranges[r_i] - ranges[r_i-1])
                interp_factor = int(interp_factor*1000)/1000

                ir = (1 - interp_factor) * h_t_tau[:, r_i - 1] + interp_factor * h_t_tau[:, r_i]
                ir = ir - np.mean(ir)

                last_distance = dists[y_i]

            y[y_i] = np.dot(x[y_i:y_i + n_samples], ir)

        return y

    def get_h_t(self,
                  source_depth: lps_qty.Distance,
                  distance: lps_qty.Distance) -> np.array:

        depth_index = bisect.bisect_left(self.depths, source_depth)

        if depth_index == len(self.depths):
            depth_index -= 1
        elif depth_index != 0 and (self.depths[depth_index] - source_depth > \
                                   source_depth - self.depths[depth_index - 1]):
            depth_index -= 1

        h_t_tau = self.h_t_tau[depth_index].T[::-1]

        ranges = [r.get_m() for r in self.ranges]

        r_i = bisect.bisect_right(ranges, distance)
        interp_factor = (distance - ranges[r_i-1])/(ranges[r_i] - ranges[r_i-1])
        interp_factor = int(interp_factor*1000)/1000

        ir = (1 - interp_factor) * h_t_tau[:, r_i - 1] + interp_factor * h_t_tau[:, r_i]

        ir = ir - np.mean(ir)

        print(r_i, ": ", interp_factor, " -> ", np.mean(ir))

        return ir

    def print_h_t_tau(self, filename: str, source_id: int = 0) -> None:
        """
        Print the transfer function h(t) as an image, with distance and time axes.

        Args:
            filename: The file path where the image should be saved.
        """
        time = [t.get_s() for t in self.times]
        plt.figure()
        for r in range(self.h_t_tau.shape[1]):
            if self.ranges[r] == lps_qty.Distance.m(0):
                continue
            plt.plot(time, self.h_t_tau[source_id, r, :], label=f"{self.ranges[r]}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend(title="Range")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # plt.imshow(abs(self.h_t_tau[source_id,:,:]), aspect='auto',
        #             cmap='jet', interpolation='none',
        #             extent=[
        #                             self.times[0].get_s(),
        #                             self.times[-1].get_s(),
        #                             self.ranges[-1].get_m(),
        #                             self.ranges[0].get_m()]
        #             )

        # plt.xlabel("Time (s)")
        # plt.ylabel("Distance (m)")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(filename)
        # plt.close()


class Model(enum.Enum):
    """ Enum class to represent available propagation models. """
    OASES = 0

    def __str__(self) -> str:
        return self.name.title()

    def estimate_transfer_function(self,
                description: lps_channel.Description,
                source_depth: typing.List[lps_qty.Distance],
                sensor_depth: lps_qty.Distance,
                max_distance: lps_qty.Distance = lps_qty.Distance.km(1),
                max_distance_points: int = 128,
                sample_frequency: lps_qty.Frequency = lps_qty.Frequency.khz(16),
                frequency_range: typing.Tuple[lps_qty.Frequency] = None,
                filename: str = "test.dat"):
        """ Function to estimate a transfer function """
        if self == Model.OASES:
            ir = ImpulseResponse(*oases.estimate_transfer_function(
                    description = description,
                    source_depth = source_depth,
                    sensor_depth = sensor_depth,
                    max_distance = max_distance,
                    max_distance_points = max_distance_points,
                    sample_frequency = sample_frequency,
                    frequency_range = frequency_range,
                    filename = filename))
            return ir

        raise NotImplementedError(f"Estimate_transfer_function not implemented for {self}")
