import os
import typing
import dataclasses

import pickle
import bisect
import numpy as np
import matplotlib.pyplot as plt
import scipy

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

        resample_ir = round(sample_frequency.get_hz()) != round(self.sample_frequency.get_hz())

        if resample_ir:
            duration = n_samples * (sample_frequency / self.sample_frequency)
            ir_samples = int(round(duration))
        else:
            ir_samples = n_samples

        x = np.concatenate((np.zeros(ir_samples-1), data))
        y = np.zeros_like(input_data)


        dists = [d.get_m() for d in distance]
        if len(input_data) != len(dists):
            dists = np.interp(np.linspace(0, 1, len(input_data)),
                              np.linspace(0, 1, len(dists)),
                              dists)
        dists = abs(dists)



        ranges = np.asarray([r.get_m() for r in self.ranges])
        n = len(ranges)

        range_indices = np.empty(len(dists), dtype=int)

        for i, d in enumerate(dists):

            r = np.clip(bisect.bisect_right(ranges, d), 1, n - 1)
            frac = (d - ranges[r - 1]) / (ranges[r] - ranges[r - 1])

            range_indices[i] = r if frac > 0.5 else r - 1

        last_r_idx = None
        ir = None

        for y_i in range(len(input_data)):

            r_idx = range_indices[y_i]

            if r_idx != last_r_idx:

                ir = h_t_tau[:, r_idx]

                if resample_ir:
                    ir = scipy.signal.resample(ir, ir_samples)

                ir = ir - np.mean(ir)
                last_r_idx = r_idx


            y[y_i] = np.dot(x[y_i:y_i + ir_samples], ir)

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

        if n_f >= (fft_samples / 2) * 0.8: # greater than 80% of the bandwidth
            window = np.ones(n_f)
        else:
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

    def print_as_image(
        self,
        filename: str | None = None,
        db: bool = True,
        cmap: str = "viridis",
    ):
        """
        Plot h_t_tau[-1, :, :] as an imagesc-like image.

        Y-axis: range [m]
        X-axis: time [s]
        """

        if self.h_t_tau is None:
            raise RuntimeError("TemporalResponse.h_t_tau is None")

        # Select last depth
        h = self.h_t_tau[-1, :, :]  # (range, time)

        if db:
            h_plot = 20 * np.log10(np.clip(np.abs(h), 1e-12, None))
            label = "Amplitude [dB]"
        else:
            h_plot = h
            label = "Amplitude"

        ranges = np.array([r.get_m() for r in self.ranges])
        times = [t.get_s() for t in self.get_time_axis()]

        extent = [
            times[0],
            times[-1],
            ranges[-1],
            ranges[0],
        ]

        plt.figure(figsize=(10, 6))
        plt.imshow(
            h_plot,
            aspect="auto",
            extent=extent,
            cmap=cmap,
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Range [m]")
        plt.colorbar(label=label)
        plt.title("Channel Impulse Response (last depth)")
        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=150)
            plt.close()
        else:
            plt.show()


def apply_doppler_by_sample(input_data: np.array,
                  speeds: typing.List[lps_qty.Speed],
                  sound_speed: lps_qty.Speed,
                  sample_frequency: lps_qty.Frequency | int,
    ) -> np.array:
    """ Applies time-varying doppler based on approach speed.

    Args:
        input_data (np.array): Input data
        speeds (typing.List[lps_qty.Speed]): The speed across the data should be equivalent to a
            block slice of input_data.
        sound_speed (lps_qty.Speed): Reference sound speed propagation
        sample_frequency (lps_qty.Frequency): Sampling frequency of the input data

    Returns:
        np.array: output data with zero padding to keep size and the number of efective samples
    """

    if isinstance(sample_frequency, lps_qty.Frequency):
        sample_frequency = int(sample_frequency.get_hz())

    n = len(input_data)
    t = np.arange(n) / sample_frequency

    num_blocks = len(speeds)
    samples_per_block = n // num_blocks

    v = np.zeros(n)
    for i, speed in enumerate(speeds):
        v[i * samples_per_block:(i + 1) * samples_per_block] = speed.get_m_s()

    c = sound_speed.get_m_s()

    dtau_dt = c / (c + v)

    tau = np.cumsum(dtau_dt) / sample_frequency
    tau -= tau[0]

    interp = scipy.interpolate.interp1d(
        tau,
        input_data[:len(tau)],
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    y = interp(t)

    return y


def apply_doppler_by_block(input_data: np.array,
                  speeds,
                  sound_speed):

    input_samples = len(input_data)
    num_blocks = len(speeds)
    samples_per_block = input_samples//num_blocks

    output = np.array([])
    for i_block in range(num_blocks):
        doppler_factor = (sound_speed + speeds[i_block]) / sound_speed

        scaled_data = scipy.signal.resample(
                input_data[i_block * samples_per_block:(i_block+1) * samples_per_block],
                int(samples_per_block//doppler_factor))

        output = np.concatenate((output, scaled_data))

    return output


def apply_doppler_block_crossfade(
    input_data: np.ndarray,
    speeds,
    sound_speed,
    sample_frequency: lps_qty.Frequency,
    fade_time: lps_qty.Time = lps_qty.Time.s(0.01),  # seconds
) -> np.ndarray:
    """
    Doppler by blocks with crossfade to avoid clicks.
    """
    from scipy.signal import resample_poly, windows

    input_samples = len(input_data)
    num_blocks = len(speeds)
    samples_per_block = input_samples // num_blocks

    fade_len = int(fade_time * sample_frequency)
    window = windows.hann(2 * fade_len)

    output = np.zeros(0)
    prev_tail = None

    for i_block in range(num_blocks):
        v = speeds[i_block].get_m_s()
        c = sound_speed.get_m_s()
        doppler_factor = (c + v) / c

        block = input_data[
            i_block * samples_per_block:(i_block + 1) * samples_per_block
        ]

        scaled = resample_poly(
            block,
            int(samples_per_block / doppler_factor),
            samples_per_block,
        )

        if prev_tail is not None:
            fade_out = prev_tail * window[:fade_len]
            fade_in = scaled[:fade_len] * window[fade_len:]
            output[-fade_len:] = fade_out + fade_in
            output = np.concatenate((output, scaled[fade_len:]))
        else:
            output = np.concatenate((output, scaled))

        prev_tail = scaled[-fade_len:]

    return output
