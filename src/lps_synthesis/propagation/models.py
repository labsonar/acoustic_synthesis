"""Module for representing the channel impulse response and methods for estimating them
"""
import os
import enum
import typing

import numpy as np
import pickle

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.channel_description as lps_channel
import lps_synthesis.propagation.oases as oases

class ImpulseResponse():
    """ Simple class to represent all the data needed to represent a response. """
    def __init__(self,
                 h_f_tau: np.array = None,
                 h_t_tau: np.array = None,
                 depths: typing.List[lps_qty.Distance] = None,
                 ranges: typing.List[lps_qty.Distance] = None,
                 frequencies: typing.List[lps_qty.Frequency] = None,
                 times: typing.List[lps_qty.Time] = None):
        self.h_f_tau = h_f_tau
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
                ir.h_f_tau, ir.h_t_tau, ir.depths, ir.ranges, ir.times, ir.frequencies \
                        = pickle.load(file)
                return ir
        return None

    def save(self, filename: str) -> None:
        """ Save a impulse response to file. """
        with open(filename, 'wb') as file:
            pickle.dump((self.h_f_tau, self.h_t_tau, self.depths,
                         self.ranges, self.times, self.frequencies), file)


class Model(enum.Enum):
    """ Enum class to represent available propagation models. """
    OASES = 0

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
            return ImpulseResponse(*oases.estimate_transfer_function(
                    description = description,
                    source_depth = source_depth,
                    sensor_depth = sensor_depth,
                    max_distance = max_distance,
                    max_distance_points = max_distance_points,
                    sample_frequency = sample_frequency,
                    frequency_range = frequency_range,
                    filename = filename))

        raise NotImplementedError(f"Estimate_transfer_function not implemented for {self}")
