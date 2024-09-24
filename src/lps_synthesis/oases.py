import typing
import numpy as np
import struct
import os

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation as lps_prop

def _ssp_to_str_list(ssp: lps_prop.SSP) -> typing.List[str]:
    ret = []

    paired_list = list(zip(ssp.depths, ssp.speeds))
    sort_idx = sorted(paired_list, key=lambda x: x[0].get_m())
    depths, speeds = zip(*sort_idx)
    depths = list(depths)
    speeds = list(speeds)

    if depths[0].get_m() != 0:
        depths.insert(0, lps_qty.Distance.m(0))
        speeds.insert(0, speeds[0])

    for depth, speed in zip(depths, speeds):
        ret.append(f"{depth.get_m():.6f} {speed.get_m_s():.6f} 0.000000 0.000000 0.000000 1.000000 0.000000")

    return ret

def _seabed_str(channel: lps_prop.AcousticalChannel) -> typing.List[str]:
    return (f"{channel.ssp.depths[-1].get_m():.6f} "
           f"{channel.bottom.get_speed(channel.interpolation).get_m_s():.6f} "
           "0.000000 0.000000 0.000000 "
           f"{channel.bottom.get_density(channel.interpolation).get_g_cm3():.6f} "
           "0.000000")


class Sweep():
    def __init__(self, start, step, n_steps) -> None:
        self.start = start
        self.step = step
        self.n_steps = n_steps

    def get_start(self):
        return self.start

    def get_step(self):
        return self.step

    def get_end(self):
        return self.start + self.step * (self.n_steps - 1)

    def get_step_value(self, step_i: int):
        return self.start + self.step * (step_i)

    def get_n_steps(self):
        return self.n_steps

    def get_all(self):
        return list([self.get_step_value(n) for n in range(self.n_steps)])



def export_dat_file(channel: lps_prop.AcousticalChannel,
                    source_depth: lps_qty.Distance,
                    sensor_depth: Sweep,
                    distance: Sweep,
                    frequency: lps_qty.Frequency,
                    filename: str):

    file_content = "LPS Syhnthesis Propagation File\n"
    file_content += "N J f\n"
    file_content += f"{frequency.get_hz()} 0\n"

    ssp_list = _ssp_to_str_list(channel.ssp)

    n_layers = len(ssp_list) + (1 if channel.bottom is not None else 0)

    if n_layers == 1:
        ssp_list.append(ssp_list[0])
        n_layers+=1

    file_content += f"{n_layers}\n"
    for layer in ssp_list:
        file_content += f"{layer}\n"

    if channel.bottom is not None:
        file_content += f"{_seabed_str(channel)}\n"


    file_content += f"{source_depth.get_m():.0f}\n"

    file_content += f"{sensor_depth.get_start().get_m():.0f} {sensor_depth.get_end().get_m():.0f} {sensor_depth.get_n_steps()}\n"

    file_content += "300.000000 1.000000e+08\n"
    file_content += "-1 0 0 0\n"

    n_samples = 1024
    time_inc = ((1.5*distance.get_end())/channel.ssp.speeds[0])/n_samples
    min_time_inc = lps_qty.Time.ms(5)
    if time_inc < min_time_inc:
        time_inc = min_time_inc
    file_content += f"{n_samples} {frequency.get_hz():.6f} {frequency.get_hz():.6f} {time_inc.get_s():.6f} {distance.get_start().get_km():.6f} {distance.get_step().get_km():.6f} {distance.get_n_steps():.0f}"

    with open(filename, 'w', encoding="utf-8") as file:
        file.write(file_content)


def trf_reader(filename):

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
        icdr = struct.unpack('i', fid.read(4))[0]  # Skip icdr

        fid.read(8)
        omegim = struct.unpack('f', fid.read(4))[0]  # Imaginary part of the radian frequency

        range_ = np.arange(r1, r1 + dr * nr, dr)  # Range values

        # Read and skip various data
        fid.read(8)  # Skips
        msuft = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        isrow = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        inttyp = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        idummy1 = struct.unpack('i', fid.read(4))[0]
        fid.read(8)
        idummy2 = struct.unpack('i', fid.read(4))[0]
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
