import typing
import abc
import dataclasses

import math
import functools
import numpy as np

import lps_utils.quantities as lps_qty
import lps_utils.hashable as lps_hash
import lps_synthesis.propagation.channel_description as lps_channel
import lps_synthesis.propagation.channel_response as lps_channel_rsp


T = typing.TypeVar("T")

@dataclasses.dataclass(frozen=True, slots=True)
class Sweep(typing.Generic[T]):
    """
    Generic sweep representation: start + i * step, for i in [0, n_steps).
    """

    start: T
    step: T
    n_steps: int

    def __len__(self) -> int:
        return self.n_steps

    def __iter__(self) -> typing.Iterator[T]:
        for i in range(self.n_steps):
            yield self[i]

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            return self._slice(idx)

        if idx < 0:
            idx += self.n_steps

        if idx < 0 or idx >= self.n_steps:
            raise IndexError("Sweep index out of range")

        return self.start + self.step * idx

    def _slice(self, s: slice) -> "Sweep[T]":

        start_i, stop_i, step_i = s.indices(self.n_steps)

        if step_i != 1:
            raise ValueError("Slice step != 1 not supported for Sweep")

        new_n = max(0, stop_i - start_i)

        return Sweep[T](
            start=self[start_i],
            step=self.step,
            n_steps=new_n
        )

    def get_end(self) -> T:
        """ Get last element """
        return self[len(self)-1]

    def to_list(self) -> list[T]:
        """ Get elements as list """
        return list(self)

    def __repr__(self):
        return f"Sweep(start={self.start}, step={self.step}, n_steps={self.n_steps})"


@dataclasses.dataclass(frozen=True)
class FrequencyGrid:
    """
    Class to represent a fragment of a fft.
    """
    frequencies: list[lps_qty.Frequency]
    df: lps_qty.Frequency
    n_fft: int
    k0: int

@dataclasses.dataclass(slots=True, eq=False)
class QueryConfig(lps_hash.Hashable):
    """
    Unified query for getting the channel response from propagation models.
    """
    description: lps_channel.Description

    sensor_depth: lps_qty.Distance
    source_depths: typing.List[lps_qty.Distance] = dataclasses.field(
        default_factory=lambda: [
            lps_qty.Distance.m(d) for d in np.arange(3, 25, 5)
        ]
    )

    max_distance: lps_qty.Distance = lps_qty.Distance.km(1)
    max_distance_points: int | None = None

    frequency_range: typing.Tuple[lps_qty.Frequency, lps_qty.Frequency] | None = None
    sample_frequency: lps_qty.Frequency = lps_qty.Frequency.khz(16)

    def get_source_sweep(self) -> Sweep[lps_qty.Distance]:
        """
        Builds a Sweep for source depths that exactly contains
        the provided depth list using the GCD strategy.
        """

        if len(self.source_depths) == 0:
            raise ValueError("source_depths cannot be empty")

        if len(self.source_depths) == 1:
            return Sweep(
                start=self.source_depths[0],
                step=lps_qty.Distance.m(1),
                n_steps=1
            )

        depths_m = np.array(sorted(int(d.get_m()) for d in self.source_depths))

        start = depths_m[0]
        stop = depths_m[-1]

        diffs = depths_m[1:] - depths_m[0]
        step = functools.reduce(math.gcd, diffs)

        n_steps = math.ceil((stop - start) / step) + 1

        return Sweep(
            start=lps_qty.Distance.m(start),
            step=lps_qty.Distance.m(step),
            n_steps=n_steps
        )

    def get_distance_sweep(self) -> Sweep[lps_qty.Distance]:
        """
        Builds a Sweep for distance using preferred steps [1, 2, 5] × 10^n
        """

        if self.max_distance_points is None:
            max_distance_points = self.max_distance / lps_qty.Distance.m(5)
        else:
            max_distance_points = self.max_distance_points


        desired_step = self.max_distance.get_m() / max_distance_points

        magnitude = 10 ** math.floor(math.log10(desired_step))

        possible_steps = [c * magnitude for c in (1, 2, 5, 10)]
        filtered = [s for s in possible_steps if s >= desired_step]

        step = min(filtered) if filtered else max(possible_steps)

        n_steps = math.ceil(self.max_distance.get_m() / step)

        return Sweep(
            start = lps_qty.Distance.m(0),
            step = lps_qty.Distance.m(step),
            n_steps = n_steps  + 1
        )

    def get_frequency_sweep(self, range_margin: float = 1.5) -> typing.Tuple[
        typing.List[lps_qty.Frequency],
        int
    ]:
        """
        Generates the FFT frequency bins consistent with the propagation
        time window. If frequency_range is provided, returns only the bins
        inside the interval.

        Returns:
            frequencies : list of Frequency (sub-band of FFT)
            n_fft       : total FFT size
        """

        base_speed = self.description.get_base_speed()

        n_samples = int(
            np.ceil((range_margin * self.max_distance / base_speed) * self.sample_frequency)
        )

        n_fft = 2 ** math.ceil(math.log2(n_samples) - 1)

        df = self.sample_frequency / n_fft
        freqs_hz = np.arange(1, n_fft//2) * df.get_hz()

        if self.frequency_range is not None:
            f_min, f_max = self.frequency_range
            f_min_hz = f_min.get_hz()
            f_max_hz = f_max.get_hz()

            mask = (freqs_hz >= f_min_hz) & (freqs_hz <= f_max_hz)

            indices = np.where(mask)[0]
            if indices.size == 0:
                raise ValueError(
                    "frequency_range does not intersect FFT frequency grid"
                )

            freqs_hz = freqs_hz[indices]

        frequencies = [lps_qty.Frequency.hz(f) for f in freqs_hz if f != 0]

        return frequencies, n_fft

    def _get_params(self):
        return {
            "sample_frequency": self.sample_frequency,
            "description": self.description,
            "source_depths": self.source_depths,
            "sensor_depth": self.sensor_depth,
            "max_distance": self.max_distance,
            "max_distance_points": self.max_distance_points,
            "frequency_range": self.frequency_range,
        }


class PropagationModel(abc.ABC):
    """ Basic abstraction to implement propagation models. """

    def __str__(self) -> str:
        return self.__class__.__name__.lower()

    @abc.abstractmethod
    def compute_frequency_response(self, query: QueryConfig) -> lps_channel_rsp.SpectralResponse:
        """ Compute the frequency response of the channel based on a propagation model """

    def compute_response(self, query: QueryConfig) -> \
        typing.Tuple[lps_channel_rsp.SpectralResponse, lps_channel_rsp.TemporalResponse]:
        """ Compute the frequency and time response of the channel based on a propagation model """

        rsp_f = self.compute_frequency_response(query=query)
        rsp_t = lps_channel_rsp.TemporalResponse.from_spectral(rsp_f)

        return rsp_f, rsp_t
