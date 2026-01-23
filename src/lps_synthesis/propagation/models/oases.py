import os
import tqdm

import numpy as np

import lps_utils.quantities as lps_qty
import lps_utils.subprocess as lps_proc

import lps_synthesis.propagation.channel_response as lps_channel_rsp
import lps_synthesis.propagation.channel_description as lps_channel
import lps_synthesis.propagation.models.interfaces as model_core


class Oases(model_core.PropagationModel):
    """
    OASES-based acoustic propagation model.
    """

    def __init__(self, workdir: str = "./channel/oases"):
        self.workdir = workdir

    def compute_frequency_response(self, query: model_core.QueryConfig) -> \
            lps_channel_rsp.SpectralResponse:
        """
        Compute the frequency-domain channel response using OASES.
        """

        os.makedirs(self.workdir, exist_ok=True)

        base_name = "oases"
        dat_file = os.path.join(self.workdir, f"{base_name}.dat")
        trf_file = os.path.join(self.workdir, f"{base_name}.trf")

        frequencies, _,  = query.get_frequency_sweep()

        with tqdm.tqdm(
            total=len(frequencies),
            desc="Frequencies",
            ncols=120,
            leave=False,
        ) as pbar:

            def on_output(line: str):
                if line.startswith("FREQ. NO."):
                    pbar.update(1)

            self._export_dat_file(query = query, filename = dat_file)

            lps_proc.run_process(
                comand=f"oasp {base_name}",
                running_directory=self.workdir,
                on_output=on_output,
            )

        # if os.path.exists(dat_file):
        #     os.remove(dat_file)
        # if os.path.exists(trf_file):
        #     os.remove(trf_file)

        return self._read_trf_file(trf_file)

    @staticmethod
    def _export_dat_file(query: model_core.QueryConfig, filename: str):
        """
        Export OASES .dat file.
        """

        def to_oases_format(description: lps_channel.Description) -> str:
            if len(description.layers) == 0:
                raise UnboundLocalError("Should not export an empty channel")

            n_layers = len(description.layers) + (1 if description.air_sea is not None else 0)
            ret = f"{n_layers}\n"
            for depth, layer in description:
                ret += (
                    f"{depth.get_m():.6f} "
                    f"{layer.get_compressional_speed().get_m_s():6f} "
                    f"{layer.get_shear_speed().get_m_s():6f} "
                    f"{layer.get_compressional_attenuation():6f} "
                    f"{layer.get_shear_attenuation():6f} "
                    f"{layer.get_density().get_g_cm3():6f} "
                    f"{layer.get_rms_roughness().get_m():6f}\n"
                )
            return ret[:-1]

        distance_sweep = query.get_distance_sweep()
        source_sweep   = query.get_source_sweep()
        frequencies, n_fft = query.get_frequency_sweep()

        f_center = frequencies[len(frequencies)//2]

        content = ""
        content += "LPS Synthesis Propagation File\n"
        content += "N J f\n"
        content += f"{f_center.get_hz():.0f} 0\n"
        content += f"{to_oases_format(query.description)}\n"
        content += f"{query.sensor_depth.get_m():.1f}\n"
        content += (
            f"{source_sweep.start.get_m():.1f} "
            f"{source_sweep.get_end().get_m():.1f} "
            f"{len(source_sweep)}\n"
        )
        content += "300.000000 1.000000e+08\n"
        content += "-1 0 0 0\n"
        content += (
            f"{n_fft} "
            f"{frequencies[0].get_hz():.6f} "
            f"{frequencies[-1].get_hz():.6f} "
            f"{(1/query.sample_frequency).get_s():.8f} "
            f"{distance_sweep.start.get_km():.6f} "
            f"{distance_sweep.step.get_km():.6f} "
            f"{len(distance_sweep):.0f}"
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _read_trf_file(filename: str) -> lps_channel_rsp.SpectralResponse:
        """
        Read OASES .trf file and return H(depth, range, freq).
        """

        root, _ = os.path.splitext(filename)
        filename = root + ".trf"

        with open(filename, "rb") as fid:

            while fid.read(1) != b"P":
                pass
            fid.seek(-1, os.SEEK_CUR)

            while fid.read(1) not in (b"+", b"-"):
                pass

            fid.read(8)
            _fc = np.fromfile(fid, dtype=np.float32, count=1)[0]

            fid.read(8)
            _sd = np.fromfile(fid, dtype=np.float32, count=1)[0]

            fid.read(8)
            z1, z2, nz = np.fromfile(fid, dtype=np.float32, count=2).tolist() + \
                         [np.fromfile(fid, dtype=np.int32, count=1)[0]]
            depths = np.linspace(z1, z2, nz)

            fid.read(8)
            r1, dr, nr = np.fromfile(fid, dtype=np.float32, count=2).tolist() + \
                         [np.fromfile(fid, dtype=np.int32, count=1)[0]]
            ranges = r1 + dr * np.arange(nr)

            fid.read(8)
            nfft, bin_low, bin_high = np.fromfile(fid, dtype=np.int32, count=3)

            dt = np.fromfile(fid, dtype=np.float32, count=1)[0]

            df = (1/dt) / nfft
            bins = np.arange(bin_low - 1, bin_high)
            freqs = bins * df

            fid.read(8)
            _ = np.fromfile(fid, dtype=np.int32, count=1)

            fid.read(8)
            _omegim = np.fromfile(fid, dtype=np.float32, count=1)

            for _ in range(10):
                fid.read(12)

            fid.read(4)

            nf = len(freqs)
            h_f_tau = np.zeros((len(depths), len(ranges), nf), dtype=np.complex128)

            for j in range(nf):
                for r in range(len(ranges)):
                    fid.read(4)
                    raw = np.fromfile(fid, dtype=np.float32, count=len(depths) * 4 - 2)
                    real = raw[0::2]
                    imag = raw[1::2]
                    h = real[0::2] + 1j * imag[0::2]
                    h_f_tau[:, r, j] = -h # corrigindo fase geral para compatibilidade com Traceo
                    fid.read(4)

        depths = [lps_qty.Distance.m(z) for z in depths]
        ranges = [lps_qty.Distance.km(r) for r in ranges]
        freqs  = [lps_qty.Frequency.hz(f) for f in freqs]

        return lps_channel_rsp.SpectralResponse(
            h_f_tau=h_f_tau,
            depths=depths,
            ranges=ranges,
            frequencies=freqs,
            sample_frequency=lps_qty.Frequency.hz(1/dt)
        )
