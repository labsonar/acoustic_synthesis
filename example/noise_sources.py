import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wav_write

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.noise_source as lps_noise
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.signal as lps_signal


def save_wav(signal: np.ndarray, fs: int, filename: str):
    """Salva o sinal como um arquivo WAV após normalizar."""

    normalized = lps_signal.Normalization.MIN_MAX_ZERO_CENTERED(signal)
    wav_signal = (normalized * 32767).astype(np.int16)
    wav_write(filename, fs, wav_signal)

def main():
    """main function of test noise_source."""

    base_dir = "./result/noise_sources"
    os.makedirs(base_dir, exist_ok=True)

    fs = lps_qty.Frequency.khz(16)

    for ship_type in [lps_noise.ShipType.RECREATIONAL, lps_noise.ShipType.CONTAINERSHIP]:
        noise_source = lps_noise.CavitationNoise(ship_type=ship_type)

        container = lps_noise.Ship(ship_id=str(ship_type), propulsion=noise_source)
        container.move(lps_qty.Time.s(1), 15)

        bb_noise, _ = noise_source.generate_broadband_noise(fs)
        mod_noise = noise_source.generate_noise(fs)

        f_bb, i_bb = lps_bb.psd(bb_noise, fs=fs.get_hz())
        f_mod, i_mod = lps_bb.psd(mod_noise, fs=fs.get_hz())

        plt.figure()
        plt.semilogy(f_bb, i_bb, label="Broadband Noise")
        plt.semilogy(f_mod, i_mod, label="Modulated Noise", linestyle='--')
        plt.xlabel("Frequência (Hz)")
        plt.ylabel("PSD (Pa²/Hz)")
        plt.title(f"PSD - {ship_type.name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(base_dir, f"psd_{ship_type.name.lower()}.png")
        plt.savefig(fig_path)
        plt.close()

        wav_bb_path = os.path.join(base_dir, f"{ship_type.name.lower()}_broadband.wav")
        wav_mod_path = os.path.join(base_dir, f"{ship_type.name.lower()}_modulated.wav")

        save_wav(bb_noise, int(fs.get_hz()), wav_bb_path)
        save_wav(mod_noise, int(fs.get_hz()), wav_mod_path)

if __name__ == "__main__":
    main()
