"""
This module provides classes for modeling and simulating acoustic wave propagation in underwater
environments.

Classes:
    Description: Handles the layers of an acoustic channel and their properties.
    Channel: Manages the computation and storage of transfer functions using the channel description
"""
import os
import typing

import numpy as np

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.models as lps_propag_model
import lps_synthesis.propagation as lps_propag

# DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".lps", "channel")
DEFAULT_DIR = os.path.join(".", "channel")

class Channel():
    """
    Represents an acoustic channel that computes transfer functions and handles data storage.
    """

    def __init__(self,
                 query: lps_propag_model.QueryConfig,
                 model: lps_propag_model.PropagationModel = None,
                 channel_dir: typing.Optional[str] = None,
                 hash_id: str = None):

        self.query = query
        self.model = model or lps_propag_model.Oases()
        self.channel_dir = channel_dir if channel_dir is not None else DEFAULT_DIR
        self.hash_id = hash_id

        os.makedirs(self.channel_dir, exist_ok=True)

        self.response = None

        if not self._load():
            self._calc()

    def _filename(self, ext: str  = ".pkl") -> str:
        return os.path.join(self.channel_dir, f"{self._get_hash()}{ext}")

    def _load(self) -> bool:
        self.response = lps_propag.TemporalResponse.load(self._filename())
        return self.response is not None

    def _calc(self) -> None:
        _, self.response = self.model.compute_response(query=self.query)
        self.response.save(self._filename())

    def _get_hash(self) -> str:
        if self.hash_id is not None:
            return self.hash_id

        return hash(self.query)

    def propagate(self,
                  input_data: np.array,
                  source_depth: lps_qty.Distance,
                  distance: typing.List[lps_qty.Distance],
                  sample_frequency: lps_qty.Frequency) -> np.array:
        """ Calculates the signal after propagation in the channel

        Args:
            input_data (np.array): Signal to propagate, should have size (n_samples,)
            source_depth (lps_qty.Distance): Depth of the source (ship's draft)
            distance (typing.List[lps_qty.Distance]): Equivalent distance for input_data.
                If the number of distances is different from the number of samples in the input
                data, the distances are interpolated.
            sample_frequency (lps_qty.Frequency): Sample frequency of the input_data

        Returns:
            np.array: signal after propagation in the channel
        """
        return self.response.apply(input_data = input_data,
                                   source_depth = source_depth,
                                   distance = distance,
                                   sample_frequency = sample_frequency)

    def get_ir(self) -> lps_propag.TemporalResponse:
        """ Return the impulse response of a channel. """
        return self.response
