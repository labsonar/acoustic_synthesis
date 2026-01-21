import os
import typing
import dataclasses

import pickle
import bisect
import numpy as np
import scipy.signal as scipy

import lps_utils.quantities as lps_qty

@dataclasses.dataclass(slots=True)
class SpectralResponse:
    """
    Frequency-domain acoustic propagation response.
    """

    h_f_tau: np.ndarray  # H_f(depth, range, frequencies)
    depths: typing.List[lps_qty.Distance]
    ranges: typing.List[lps_qty.Distance]
    frequencies: typing.List[lps_qty.Frequency]
    sample_frequency: lps_qty.Frequency

@dataclasses.dataclass(slots=True)
class TemporalResponse:
    """Represents a time-domain acoustic impulse response."""

    h_t_tau: np.ndarray | None = None # h_t(depth, range, times)
    depths: typing.List[lps_qty.Distance] | None = None
    ranges: typing.List[lps_qty.Distance] | None = None
    sample_frequency: lps_qty.Frequency | None = None

    @classmethod
    def load(cls, filename: str) -> 'TemporalResponse':
        """ Load a impulse response from file. """
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                return cls(**pickle.load(file))
        return None

    def save(self, filename: str) -> None:
        """Save an impulse response to file."""
        with open(filename, "wb") as file:
            pickle.dump(dataclasses.asdict(self), file)

    def apply(self,
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

        depth_index = bisect.bisect_left(self.depths, source_depth)

        if depth_index == len(self.depths):
            depth_index -= 1
        elif depth_index != 0 and (self.depths[depth_index] - source_depth > \
                                   source_depth - self.depths[depth_index - 1]):
            depth_index -= 1

        h_t_tau = self.h_t_tau[depth_index].T[::-1]

        n_samples = h_t_tau.shape[0]


        if input_data.ndim == 1:
            data = input_data
        elif input_data.ndim == 2 and input_data.shape[1] == 1:
            data = input_data[:, 0]  # Flatten para vetor 1D
        else:
            raise ValueError(f"input_data must be shape (N,) or (N,1), got {input_data.shape}")

        x = np.concatenate((np.zeros(n_samples-1), data))
        y = np.zeros_like(input_data)

        dists = [d.get_m() for d in distance]
        if len(input_data) != len(dists):
            dists = np.interp(np.linspace(0, 1, len(input_data)),
                              np.linspace(0, 1, len(dists)),
                              dists)
        dists = abs(dists)

        ranges = [r.get_m() for r in self.ranges]

        last_distance = None
        ir = None
        outside_range_limits = False

        resample_ir = sample_frequency != self.sample_frequency

        for y_i in range(len(input_data)):
            if last_distance != dists[y_i]:

                if dists[y_i] < ranges[0]:
                    ir = h_t_tau[:, 0]

                    if not outside_range_limits:
                        print(f"Warning: {dists[y_i]} below the {ranges[0]} limit.")
                        outside_range_limits = True

                elif dists[y_i] > ranges[-1]:
                    ir = h_t_tau[:, -1]

                    if not outside_range_limits:
                        print(f"Warning: {dists[y_i]} above the {ranges[-1]} limit.")
                        outside_range_limits = True

                else:

                    r_i = bisect.bisect_right(ranges, dists[y_i])

                    interp_factor = (dists[y_i] - ranges[r_i-1])/(ranges[r_i] - ranges[r_i-1])

                    if interp_factor > 0.5:
                        ir = h_t_tau[:, r_i]
                    else:
                        ir = h_t_tau[:, r_i - 1]


                    # interp_factor = int(interp_factor*1000)/1000

                    # ir = (1 - interp_factor) * h_t_tau[:, r_i - 1]
                    #  + interp_factor * h_t_tau[:, r_i]

                ir = ir - np.mean(ir)

                if resample_ir:
                    duration = len(ir) * (sample_frequency/self.sample_frequency)
                    ir_samples = int(round(duration))

                    ir = scipy.resample(ir, ir_samples)

                last_distance = dists[y_i]

            y[y_i] = np.dot(x[y_i:y_i + n_samples], ir)

        return y

    @staticmethod
    def from_spectral(sr: SpectralResponse) -> "TemporalResponse":
        """
        Convert a SpectralResponse (frequency domain) into a TemporalResponse (time domain).
        """

        h_f_tau = sr.h_f_tau
        freqs = np.array([f.get_hz() for f in sr.frequencies])

        fs = sr.sample_frequency.get_hz()
        df = freqs[1] - freqs[0]
        fft_samples = int(round(fs / df))

        n_depths, n_ranges, n_f = h_f_tau.shape
        k0 = int(round(freqs[0] / df))

        window = np.hanning(n_f)

        h_t_tau = np.zeros((n_depths, n_ranges, fft_samples), dtype=np.float64)

        for d in range(n_depths):
            for r in range(n_ranges):

                sub_h_f = h_f_tau[d, r, :] * window

                h_f = np.zeros(fft_samples, dtype=np.complex128)
                h_f[k0:k0 + n_f] = sub_h_f
                h_f[fft_samples//2+1:] = np.flip(np.conj(h_f[1:fft_samples//2]))
                h_f *= fft_samples / 2

                h_t = np.fft.ifft(h_f)

                h_t_tau[d, r, :] = np.real(h_t)

        return TemporalResponse(
            h_t_tau=h_t_tau,
            depths=sr.depths,
            ranges=sr.ranges,
            sample_frequency=sr.sample_frequency
        )

    def get_time_axis(self) -> typing.List[lps_qty.Time]:
        n_samples = self.h_t_tau.shape[2]
        return [i/self.sample_frequency for i in np.arange(n_samples)]
