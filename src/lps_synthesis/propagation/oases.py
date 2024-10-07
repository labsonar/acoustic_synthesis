""" Module to acess the oases for getting impulse response of a channel
"""
import struct
import os
import glob
import subprocess

import tqdm
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
                    source_depth: lps_qty.Distance,
                    sensor_depth: Sweep,
                    distance: Sweep,
                    frequency: lps_qty.Frequency,
                    filename: str):
    """ Function to export a .dat file to call oasp """

    file_content = "LPS Syhnthesis Propagation File\n"
    file_content += "N J f\n"
    file_content += f"{frequency.get_hz()} 0\n"

    file_content += f"{description.to_oases_format()}\n"

    file_content += f"{source_depth.get_m():.0f}\n"
    file_content += (f"{sensor_depth.get_start().get_m():.0f} {sensor_depth.get_end().get_m():.0f}"
                    f" {sensor_depth.get_n_steps()}\n")

    file_content += "300.000000 1.000000e+08\n"
    file_content += "-1 0 0 0\n"

    time_inc = 0.5/frequency
    n_samples = int(np.ceil(((1.5*distance.get_end())/description.get_base_speed())/time_inc))

    file_content += (f"{n_samples} {frequency.get_hz():.6f} {frequency.get_hz():.6f}"
                    f"{time_inc.get_s():.6f} {distance.get_start().get_km():.6f}"
                    f"{distance.get_step().get_km():.6f} {distance.get_n_steps():.0f}")

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
        fc = struct.unpack('f', fid.read(4))[0]

        fid.read(8)
        sd = struct.unpack('f', fid.read(4))[0]  # Source depth

        fid.read(8)
        z1 = struct.unpack('f', fid.read(4))[0]  # First receiver depth
        z2 = struct.unpack('f', fid.read(4))[0]  # Last receiver depth
        num_z = struct.unpack('i', fid.read(4))[0]  # Number of receiver depths
        z = np.linspace(z1, z2, num_z)  # Generate depths array

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
        omegim = struct.unpack('f', fid.read(4))[0]  # Imaginary part of the radian frequency

        range_ = np.arange(r1, r1 + dr * nr, dr)  # Range values

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
        out = np.zeros((num_z, nr, nf), dtype=np.complex_)

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
                out[:, jj, j] = temp_complex[:num_z]

    return out, sd, z, range_, f, fc, omegim, dt

def estimate_transfer_function(description: lps_channel.Description,
                    source_depth: lps_qty.Distance,
                    sensor_depth: lps_qty.Distance,
                    max_distance: lps_qty.Distance = lps_qty.Distance.km(1),
                    distance_points: int = 128,
                    sample_frequency: lps_qty.Frequency = lps_qty.Frequency.khz(16),
                    n_fft: int = 128,
                    filename: str = "test.dat"):
    """ Function to estimate a transfer function """

    file_without_extension = os.path.splitext(filename)[0]

    freq_step = (sample_frequency/2)/n_fft

    frequencies = Sweep(start=freq_step, step=freq_step, n_steps=n_fft)
    depths = Sweep(start=sensor_depth, step=lps_qty.Distance.m(1), n_steps=1)
    ranges = Sweep(start=lps_qty.Distance.m(0),
                   step=max_distance/(distance_points),
                   n_steps=distance_points)

    h_f = np.zeros((frequencies.get_n_steps(), ranges.get_n_steps()), dtype=np.complex_)

    for f_i, frequency in enumerate(tqdm.tqdm(frequencies.get_all())):

        export_dat_file(description=description,
                frequency=frequency,
                source_depth = source_depth,
                sensor_depth = depths,
                distance = ranges,
                filename=filename)

        comando = f"oasp {file_without_extension}"
        resultado_str = subprocess.run(comando,
                                        shell=True,
                                        capture_output=True,
                                        text=True,
                                        check=True)

        out, _, _, _, _, _, _, _ = trf_reader(filename)
        out = out.ravel()

        if len(out) != h_f.shape[1]:
            print("####################")
            print(resultado_str.stdout)
            print("####################")
            print("Error on frequency: ", frequency)
            print("Len: ", len(out), "\texpected len: ", h_f.shape[1])
            raise UnboundLocalError("Error in Oases simulation")

        h_f[f_i, :] = out


    ts = 1/frequencies.get_end()
    t = np.arange(0, frequencies.get_n_steps() * ts.get_s(), ts.get_s())

    h_t = np.zeros((len(t), ranges.get_n_steps()), dtype=np.complex_)

    for r_i, _ in enumerate(ranges.get_all()):
        # h_t[:,r_i] = np.fft.ifft(h_f[:, r_i], len(t))
        h_t[:,r_i] = (len(t) / np.sqrt(2)) * (np.fft.ifft(h_f[:, r_i], len(t)) * \
                                              np.exp(-1j * 2 * np.pi * freq_step.get_hz() * t))

    for file in glob.glob(f"{file_without_extension}.*"):
        os.remove(file)
        print(f"Removido: {file}")

    return h_f, h_t, ranges.get_all(), t, frequencies.get_all()
