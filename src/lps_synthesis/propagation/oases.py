""" Module to acess the oases for getting impulse response of a channel
"""
import struct
import typing
import os
import math
import functools
import numpy as np

import lps_utils.quantities as lps_qty
import lps_utils.subprocess as lps_proc
import lps_synthesis.propagation.channel_description as lps_channel

class Sweep():
    """ Class to represent a sweep. """

    def __init__(self, start, step, n_steps) -> None:
        self.start = start
        self.step = step
        self.n_steps = n_steps

    def get_start(self):
        """ Get start of the sweep. """
        return self.start

    def get_step(self):
        """ Get step of the sweep. """
        return self.step

    def get_end(self):
        """ Get end of the sweep. """
        return self.start + self.step * (self.n_steps - 1)

    def get_step_value(self, step_i: int):
        """ Getting value of step_i step of the sweep . """
        return self.start + self.step * (step_i)

    def get_n_steps(self):
        """ Get number of step on the sweep. """
        return self.n_steps

    def get_all(self):
        """ Get all values on the sweep. """
        return list([self.get_step_value(n) for n in range(self.n_steps)])

    def __iter__(self):
        for i in range(self.n_steps):
            yield self.get_step_value(i)

def export_dat_file(description: lps_channel.Description,
                    source_depths: Sweep,
                    sensor_depth: lps_qty.Distance,
                    distance: Sweep,
                    sample_frequency: lps_qty.Frequency,
                    filename: str,
                    frequency_range: typing.Tuple[lps_qty.Frequency] = None):
    """ Function to export a .dat file to call oasp """

    if frequency_range is not None:
        lower_freq = frequency_range[0]
        upper_freq = frequency_range[1]
        middle_freq = (frequency_range[0] + frequency_range[1])/2
    else:
        lower_freq = lps_qty.Frequency.hz(0)
        upper_freq = sample_frequency/2
        middle_freq = sample_frequency/4

    file_content = "LPS Syhnthesis Propagation File\n"
    file_content += "N J f\n"
    file_content += f"{middle_freq.get_hz():.0f} 0\n"

    file_content += f"{description.to_oases_format()}\n"

    file_content += f"{sensor_depth.get_m():.0f}\n"
    file_content += (f"{source_depths.get_start().get_m():.0f}"
                     f" {source_depths.get_end().get_m():.0f}"
                     f" {source_depths.get_n_steps()}\n")

    file_content += "300.000000 1.000000e+08\n"
    file_content += "-1 0 0 0\n"

    time_step = 1/sample_frequency
    n_samples = int(np.ceil(((1.5*distance.get_end())/description.get_base_speed())/time_step))
    n_samples = 2 ** math.ceil(math.log2(n_samples))


    file_content += (f"{n_samples} {lower_freq.get_hz():.6f} {upper_freq.get_hz():.6f} "
                    f"{time_step.get_s():.8f} {distance.get_start().get_km():.6f} "
                    f"{distance.get_step().get_km():.6f} {distance.get_n_steps():.0f}")

    # print(file_content)
    with open(filename, 'w', encoding="utf-8") as file:
        file.write(file_content)

def trf_reader(filename):
    """ Function to read the .trf file output of a oasp """

    root, ext = os.path.splitext(filename)
    if ext.lower() != ".trf":
        filename = root + ".trf"

    with open(filename, "rb") as fid:
        byte = b" "
        while byte[0] != ord(b"P"):
            byte = fid.read(1)

        fid.seek(-1, os.SEEK_CUR)

        sign = b" "
        while sign not in (b"+", b"-"):
            sign = fid.read(1)

        fid.read(8)
        fc = struct.unpack("f", fid.read(4))[0]

        fid.read(8)
        sd = struct.unpack("f", fid.read(4))[0]

        fid.read(8)
        z1   = struct.unpack("f", fid.read(4))[0]
        z2   = struct.unpack("f", fid.read(4))[0]
        num_z = struct.unpack("i", fid.read(4))[0]
        z = np.linspace(z1, z2, num_z, dtype=np.float32)

        fid.read(8)
        r1   = struct.unpack("f", fid.read(4))[0]
        dr   = struct.unpack("f", fid.read(4))[0]
        nr   = struct.unpack("i", fid.read(4))[0]
        ranges = r1 + dr * np.arange(nr, dtype=np.float32)

        fid.read(8)
        nfft     = struct.unpack("i", fid.read(4))[0]
        bin_low  = struct.unpack("i", fid.read(4))[0]
        bin_high = struct.unpack("i", fid.read(4))[0]

        dt = struct.unpack("f", fid.read(4))[0]

        f = np.linspace(0.0, 1.0 / dt, nfft, dtype=np.float32)[(bin_low-1): bin_high]

        fid.read(8)
        _icdr = struct.unpack("i", fid.read(4))[0]

        fid.read(8)
        omegim = struct.unpack("f", fid.read(4))[0]

        for _ in range(10):
            fid.read(8)
            fid.read(4)

        fid.read(4)

        nf   = len(f)
        h_f_tau  = np.zeros((num_z, nr, nf), dtype=np.complex128)

        for j in range(nf):
            for jj in range(nr):
                fid.read(4)
                raw = np.fromfile(fid, dtype=np.float32, count=num_z * 4 - 2)

                real = raw[0::2]
                imag = raw[1::2]
                temp = real + 1j * imag
                temp = temp[0::2]

                fid.read(4)
                h_f_tau[:, jj, j] = temp

    return h_f_tau, sd, z, ranges, f, fc, omegim, dt

def trf_time_series_tranform(h_f_tau: np.ndarray,
                        freqs: np.ndarray,
                        fo: float,
                        omegim: float,
                        times: np.ndarray,
                        bw: float) -> np.ndarray:
    """
    Converts a frequency response to a time-domain response using the same
    procedure as the Octave script (Gaussian window + exponential decay).

    Parameters:
        h_f_tau : np.ndarray (depth, range, freq)
                  Complex frequency response
        freqs   : np.ndarray (freq,)
                  Frequency values (Hz) used in 'out'
        fo      : float
                  Center frequency (Hz)
        omegim  : float
                  Imaginary part of angular frequency
        times   : np.ndarray (time,)
                  Desired time vector (s)
        bw      : float
                  Bandwidth (Hz)

    Returns:
        ts_out  : np.ndarray (depth, range, time)
                  Time-domain response (range-depth time series)
    """
    f = freqs[:, None]
    t = times[None, :]

    # shape: (F, T)
    kernel = (np.exp(1j * (2*np.pi * f + 1j * omegim) * t) *
              (np.exp(-((f - fo)**2) / (2 * bw**2)) *
               np.sqrt(2 * np.pi) / len(times)))

    depth, rng, _ = h_f_tau.shape
    h_t_tau = np.empty((depth, rng, len(times)), dtype=np.complex128)

    for j in range(depth):
        temp = h_f_tau[j, :, :]
        h_t_tau[j] = temp @ kernel

    return h_t_tau

def trf_impulse_response_reader(filename: str):
    """ Function to read a .trf file and convert to impulse response in time. """

    h_f_tau, _, depths, ranges, freqs, _, omegim, dt = trf_reader(filename)

    fs = 1.0 / dt
    df = freqs[1] - freqs[0]
    n_samples = int(fs / df) + 1
    times = np.arange(0, n_samples * dt, dt)

    fo = freqs[len(freqs) // 2]
    bw = freqs[-1] - freqs[0]

    h_t_tau = trf_time_series_tranform(
        h_f_tau    = h_f_tau,
        freqs  = freqs,
        fo     = fo,
        omegim = omegim,
        times  = times,
        bw     = bw
    )

    h_t_tau = np.abs(h_t_tau)

    return h_t_tau, \
            [lps_qty.Distance.m(f) for f in depths], \
            [lps_qty.Distance.km(f) for f in ranges], \
            [lps_qty.Frequency.hz(f) for f in freqs], \
            [lps_qty.Time.s(t) for t in times]

def estimate_transfer_function(description: lps_channel.Description,
                    source_depth: typing.List[lps_qty.Distance],
                    sensor_depth: lps_qty.Distance,
                    max_distance: lps_qty.Distance = lps_qty.Distance.km(1),
                    max_distance_points: int = 128,
                    sample_frequency: lps_qty.Frequency = lps_qty.Frequency.khz(16),
                    frequency_range: typing.Tuple[lps_qty.Frequency] = None,
                    filename: str = "test"):
    """ Function to estimate a transfer function """
    original_directory = os.getcwd()

    filename = f"{filename}.dat"

    # file_without_extension = os.path.splitext(filename)[0]
    file_directory = os.path.dirname(filename)
    file_without_extension = os.path.splitext(os.path.basename(filename))[0]

    # encontra um sweep para atender a lista de distancias com base no mdc da lista
    if len(source_depth) > 1:
        depths = np.array(sorted([int(s.get_m()) for s in source_depth]))
        start = depths[0]
        stop = depths[-1]
        depths = depths[1:] - depths[0]
        step = functools.reduce(math.gcd, depths)
        n_steps = math.ceil((stop-start)/step) + 1
        depths = Sweep(start=lps_qty.Distance.m(start),
                        step=lps_qty.Distance.m(step),
                        n_steps=n_steps)
    else:
        depths = Sweep(start=source_depth[0], step=lps_qty.Distance.m(1), n_steps=1)

    # encontra um sweep para ter um step com numero preferenciais [1,2,5]
    # proximo a distancia/distance_points
    step_inicial = max_distance.get_m() / max_distance_points
    ordem_magnitude = 10 ** math.floor(math.log10(step_inicial))
    possible_steps = [c * ordem_magnitude for c in [1, 2, 5, 10]]
    filt_steps = [step for step in possible_steps if step > step_inicial]
    step = min(filt_steps) if filt_steps else max(possible_steps)
    # step = min(possible_steps, key=lambda x: abs(x - step_inicial))
    n_steps = math.ceil(max_distance.get_m()/step) + 1
    ranges = Sweep(start=lps_qty.Distance.m(0),
                   step=lps_qty.Distance.m(step),
                   n_steps=n_steps)

    export_dat_file(description=description,
            sample_frequency=sample_frequency,
            source_depths = depths,
            sensor_depth = sensor_depth,
            distance = ranges,
            filename=filename,
            frequency_range=frequency_range)

    comand = f"oasp {file_without_extension}"

    lps_proc.run_process(comand=comand,
                         running_directory=file_directory if file_directory else None)

    return trf_impulse_response_reader(filename)
