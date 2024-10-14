""" Module to acess the oases for getting impulse response of a channel
"""
import struct
import typing
import os
import glob
import subprocess
import math
import functools
import numpy as np

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.acoustical_channel as lps_channel

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
        lower_freq = lps_qty.Frequency.hz(10)
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

    # check corret extension
    root, ext = os.path.splitext(filename)
    if ext != '.trf':
        filename = root + '.trf'

    with open(filename, 'rb') as fid:

        junk = b' '
        while junk[0] != ord(b'P'):
            junk = fid.read(1)

        # Move back the file pointer to the start of 'P'
        fid.seek(-1, 1)

        # Read the first sign ('+' or '-')
        sign = b' '
        while sign not in [b'+', b'-']:
            sign = fid.read(1)

        # Skip next 2 floats and read center frequency
        fid.read(8)
        _ = struct.unpack('f', fid.read(4))[0]

        fid.read(8)
        _ = struct.unpack('f', fid.read(4))[0]  # Source depth

        fid.read(8)
        z1 = struct.unpack('f', fid.read(4))[0]  # First receiver depth
        z2 = struct.unpack('f', fid.read(4))[0]  # Last receiver depth
        num_z = struct.unpack('i', fid.read(4))[0]  # Number of receiver depths
        _ = np.linspace(z1, z2, num_z)  # Generate depths array

        fid.read(8)
        r1 = struct.unpack('f', fid.read(4))[0]  # First receiver range
        dr = struct.unpack('f', fid.read(4))[0]  # Range increment
        nr = struct.unpack('i', fid.read(4))[0]  # Number of ranges

        fid.read(8)
        nfft = struct.unpack('i', fid.read(4))[0]  # FFT size
        bin_low = struct.unpack('i', fid.read(4))[0]  # Low frequency bin
        bin_high = struct.unpack('i', fid.read(4))[0]  # High frequency bin

        dt = struct.unpack('f', fid.read(4))[0]  # Sampling interval
        f = np.linspace(0, 1/dt, nfft)[:nfft]  # Frequencies
        f = f[bin_low:bin_high + 1]  # Keep only the bins between bin_low and bin_high

        fid.read(8)
        _ = struct.unpack('i', fid.read(4))[0]  # Skip icdr

        fid.read(8)
        _ = struct.unpack('f', fid.read(4))[0]  # Imaginary part of the radian frequency

        _ = np.arange(r1, r1 + dr * nr, dr)  # Range values

        # Read and skip various data
        fid.read(8)  # Skips
        _ = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        _ = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        _ = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        _ = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        _ = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        dummy1 = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        dummy2 = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        dummy3 = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        dummy4 = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        dummy5 = struct.unpack('i', fid.read(4))[0]
        fid.read(4)  # Skips final float

        # Initialize output array
        nf = len(f)
        h_f = np.zeros((num_z, nr, nf), dtype=np.complex_)

        # Read complex transfer function data
        for j in range(nf):
            for jj in range(nr):
                fid.read(4)  # Skip
                temp = np.fromfile(fid, dtype='float32', count=num_z * 4 - 2)
                real_part = temp[0::2]  # Real part
                imag_part = temp[1::2]  # Imaginary part
                temp_complex = real_part + 1j * imag_part  # Combine into complex numbers
                temp_complex = temp_complex[0::2]
                fid.read(4)  # Skip
                h_f[:, jj, j] = temp_complex[:num_z]

    return h_f, f, dt

def estimate_transfer_function(description: lps_channel.Description,
                    source_depth: typing.List[lps_qty.Distance],
                    sensor_depth: lps_qty.Distance,
                    max_distance: lps_qty.Distance = lps_qty.Distance.km(1),
                    max_distance_points: int = 128,
                    sample_frequency: lps_qty.Frequency = lps_qty.Frequency.khz(16),
                    frequency_range: typing.Tuple[lps_qty.Frequency] = None,
                    filename: str = "test.dat"):
    """ Function to estimate a transfer function """

    file_without_extension = os.path.splitext(filename)[0]

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

    comando = f"oasp {file_without_extension}"

    process = subprocess.Popen(comando, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    stderr_output = process.stderr.read()
    if stderr_output:
        raise UnboundLocalError(f"Erro: {stderr_output.strip()}")

    h_freqs, freqs, dt = trf_reader(filename)

    fs = 1 / dt
    df = freqs[1] - freqs[0]

    frequencies = np.arange(0, (fs+df), df)
    frequencies = np.round(frequencies * 1000) / 1000
    freqs = np.round(freqs * 1000) / 1000

    n_samples = len(frequencies)
    times = np.arange(0, n_samples * dt, dt)

    h_f_tau = np.zeros((h_freqs.shape[0], h_freqs.shape[1], n_samples), dtype=np.complex_)
    h_t_tau = np.zeros((h_freqs.shape[0], h_freqs.shape[1], n_samples), dtype=np.complex_)

    indexes = np.isin(frequencies, freqs).nonzero()[0]

    for d in range(h_freqs.shape[0]): # depths
        for r in range(h_freqs.shape[1]):# ranges
            h_f_tau[d, r, indexes] = h_freqs[d, r, :]
            h_t_tau[d, r, :] = len(times) / np.sqrt(2) * np.fft.ifft(h_f_tau[d, r, :])

    for file in glob.glob(f"{file_without_extension}.*"):
        os.remove(file)
        print(f"Removido: {file}")

    return h_f_tau, h_t_tau, \
            depths.get_all(), ranges.get_all(), \
            [lps_qty.Frequency.hz(f) for f in frequencies], \
            [lps_qty.Time.s(t) for t in times]
