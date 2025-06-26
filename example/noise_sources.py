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

def plot_spectrogram(signal, fs_hz, title, filename):
    plt.figure()
    plt.specgram(signal, NFFT=1024, Fs=fs_hz, noverlap=512, cmap='viridis')
    plt.title(title)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Frequência (Hz)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

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

    brownian_noise = lps_noise.NarrowBandNoise.with_brownian_modulation(
                                                    frequency=lps_qty.Frequency.khz(2),
                                                    amp_db_p_upa=150,
                                                    amp_std = 0,
                                                    phase_std = 0.3)

    am_noise = lps_noise.NarrowBandNoise.with_sine_am_modulation(
                                                    frequency=lps_qty.Frequency.khz(2),
                                                    amp_db_p_upa=150,
                                                    am_freq=lps_qty.Frequency.hz(1),
                                                    am_depth=0.4)

    fm_noise = lps_noise.NarrowBandNoise.with_sine_fm_modulation(
                                                    frequency = lps_qty.Frequency.khz(2),
                                                    amp_db_p_upa = 150,
                                                    oscilation_freq = lps_qty.Frequency.hz(1),
                                                    deviation_freq = lps_qty.Frequency.khz(1))

    fm_chirp = lps_noise.NarrowBandNoise.with_fm_chirp(
                                                    amp_db_p_upa = 150,
                                                    start_frequency = lps_qty.Frequency.khz(2),
                                                    end_frequency = lps_qty.Frequency.khz(4),
                                                    tx_interval = lps_qty.Time.s(1),
                                                    tx_duration = lps_qty.Time.s(0.8))

    container = lps_noise.NoiseContainer("")
    container.add_source(brownian_noise)
    container.add_source(am_noise)
    container.add_source(fm_noise)
    container.add_source(fm_chirp)
    container.move(lps_qty.Time.s(1), 15)

    brownian = brownian_noise.generate_noise(fs)
    am = am_noise.generate_noise(fs)
    fm = fm_noise.generate_noise(fs)
    chirp = fm_chirp.generate_noise(fs)

    save_wav(brownian, int(fs.get_hz()), os.path.join(base_dir, "narrowband_brownian.wav"))
    save_wav(am, int(fs.get_hz()), os.path.join(base_dir, "narrowband_am.wav"))
    save_wav(fm, int(fs.get_hz()), os.path.join(base_dir, "narrowband_fm.wav"))
    save_wav(chirp, int(fs.get_hz()), os.path.join(base_dir, "narrowband_chirp.wav"))

    plot_spectrogram(brownian, fs.get_hz(),
                     "Spectrogram - Narrowband Brownian",
                     os.path.join(base_dir, "spec_narrowband_brownian.png"))

    plot_spectrogram(am, fs.get_hz(),
                     "Spectrogram - Narrowband AM",
                     os.path.join(base_dir, "spec_narrowband_am.png"))

    plot_spectrogram(fm, fs.get_hz(),
                     "Spectrogram - Narrowband FM",
                     os.path.join(base_dir, "spec_narrowband_fm.png"))

    plot_spectrogram(chirp, fs.get_hz(),
                     "Spectrogram - Narrowband FM chirp",
                     os.path.join(base_dir, "spec_narrowband_chirp.png"))

if __name__ == "__main__":
    main()
