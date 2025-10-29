import os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tikz
import scipy.io.wavfile as wav_file
import scipy.signal as scipy

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.noise_source as lps_noise
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.signal as lps_signal

def _plot_spectrogram(signal, fs_hz, filename: str):
    f, t, Sxx = scipy.spectrogram(signal, fs=fs_hz, nperseg=4096, noverlap=2048)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    plt.figure()
    plt.imshow(Sxx_dB.T,
            origin='upper',
            aspect='auto',
            extent=[f[0], f[-1], t[-1], t[0]],
            cmap='jet')
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Tempo (s)")
    plt.colorbar(label="dB")
    plt.tight_layout()

    if filename.endswith('.tex'):
        tikz.save(filename)
    else:
        plt.savefig(filename)
    plt.close()

def _plot_psd(bb_noise, modulated_noise, fs, name, base_dir):

    f_bb, i_bb = lps_bb.psd(bb_noise, fs=fs.get_hz())
    f_mod, i_mod = lps_bb.psd(modulated_noise, fs=fs.get_hz())

    plt.figure()
    plt.semilogy(f_bb, i_bb, label="Broadband Noise")
    plt.semilogy(f_mod, i_mod, label="Modulated Noise", linestyle='--')
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("PSD (Pa²/Hz)")
    plt.title(f"PSD - {name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(base_dir, f"psd_{name.lower()}.png")
    plt.savefig(fig_path)
    plt.close()

def _plot_time_signals(bb_noise, modulating_noise, modulated_noise, fs, name, base_dir):
    t = np.arange(len(bb_noise)) / fs.get_hz()

    show_interval = lps_qty.Time.s(3)
    samples = range(int(min(show_interval * fs,len(t))))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t[samples], modulating_noise[samples])
    plt.title('Sinal de Banda Estreita (Soma dos Harmônicos)')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(t[samples], bb_noise[samples])
    plt.title('Ruído de Banda Larga (Gaussiano)')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(t[samples], modulated_noise[samples])
    plt.title('Sinal Modulado (Banda Estreita + Ruído)')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    time_fig_path = os.path.join(base_dir, f"time_signals_{name.lower()}.png")
    plt.savefig(time_fig_path)
    plt.close()

def main():
    """main function of test noise_source."""

    base_dir = "./result/noise_sources"
    os.makedirs(base_dir, exist_ok=True)

    fs = lps_qty.Frequency.khz(16)

    # for ship_type in [lps_noise.ShipType.RECREATIONAL, lps_noise.ShipType.CONTAINERSHIP]:
    #     noise_source = lps_noise.CavitationNoise(ship_type=ship_type)

    #     container = lps_noise.Ship(ship_id=str(ship_type), propulsion=noise_source)
    #     container.move(lps_qty.Time.s(1), 15)

    #     bb_noise, speeds = noise_source.generate_broadband_noise(fs)
    #     modulated_noise, modulating_noise = noise_source.modulate_noise(broadband=bb_noise,
    #                                                                     speeds=speeds,
    #                                                                     fs=fs)

    #     _plot_psd(bb_noise, modulated_noise, fs, ship_type.name, base_dir)
    #     _plot_time_signals(bb_noise, modulating_noise, modulated_noise, fs,
    #                       ship_type.name, base_dir)

    #     wav_bb_path = os.path.join(base_dir, f"{ship_type.name.lower()}_broadband.wav")
    #     wav_mod_path = os.path.join(base_dir, f"{ship_type.name.lower()}_modulated.wav")

    #     lps_signal.save_normalized_wav(bb_noise, fs, wav_bb_path)
    #     lps_signal.save_normalized_wav(modulated_noise, fs, wav_mod_path)

    sin_noise = lps_noise.NarrowBandNoise(frequency=lps_qty.Frequency.khz(2),
                                               amp_db_p_upa=150)

    brownian_noise = lps_noise.NarrowBandNoise.with_brownian_modulation(
                                                    frequency=lps_qty.Frequency.khz(2),
                                                    amp_db_p_upa=150,
                                                    amp_std = 0.1,
                                                    phase_std = 0.05)

    brownian_noise2 = lps_noise.NarrowBandNoise.with_brownian_modulation(
                                                    frequency=lps_qty.Frequency.khz(2),
                                                    amp_db_p_upa=150,
                                                    amp_std = 1e-3,
                                                    phase_std = 0.2)

    am_noise = lps_noise.NarrowBandNoise.with_sine_am_modulation(
                                                    frequency=lps_qty.Frequency.khz(2),
                                                    amp_db_p_upa=150,
                                                    am_freq=lps_qty.Frequency.hz(0.5),
                                                    am_depth=1)

    fm_noise = lps_noise.NarrowBandNoise.with_sine_fm_modulation(
                                                    frequency = lps_qty.Frequency.khz(2),
                                                    amp_db_p_upa = 150,
                                                    oscilation_freq = lps_qty.Frequency.hz(0.5),
                                                    deviation_freq = lps_qty.Frequency.khz(1))

    fm_chirp = lps_noise.NarrowBandNoise.with_fm_chirp(
                                                    amp_db_p_upa = 150,
                                                    start_frequency = lps_qty.Frequency.khz(2),
                                                    end_frequency = lps_qty.Frequency.khz(4),
                                                    tx_interval = lps_qty.Time.s(2),
                                                    tx_duration = lps_qty.Time.s(0.8))

    container = lps_noise.NoiseContainer("")
    container.add_source(sin_noise)
    container.add_source(brownian_noise)
    container.add_source(brownian_noise2)
    container.add_source(am_noise)
    container.add_source(fm_noise)
    container.add_source(fm_chirp)
    container.move(lps_qty.Time.s(1), 8)

    sin = sin_noise.generate_noise(fs)
    brownian = brownian_noise.generate_noise(fs)
    brownian2= brownian_noise2.generate_noise(fs)
    am = am_noise.generate_noise(fs)
    fm = fm_noise.generate_noise(fs)
    chirp = fm_chirp.generate_noise(fs)

    lps_signal.save_normalized_wav(sin,
                                   fs,
                                   os.path.join(base_dir, "narrowband_sin.wav"))
    lps_signal.save_normalized_wav(brownian,
                                   fs,
                                   os.path.join(base_dir, "narrowband_brownian.wav"))
    lps_signal.save_normalized_wav(brownian2,
                                   fs,
                                   os.path.join(base_dir, "narrowband_brownian2.wav"))
    lps_signal.save_normalized_wav(am,
                                   fs,
                                   os.path.join(base_dir, "narrowband_am.wav"))
    lps_signal.save_normalized_wav(fm,
                                   fs,
                                   os.path.join(base_dir, "narrowband_fm.wav"))
    lps_signal.save_normalized_wav(chirp,
                                   fs,
                                   os.path.join(base_dir, "narrowband_chirp.wav"))

    _plot_spectrogram(sin, fs.get_hz(),
                     os.path.join(base_dir, "spec_narrowband_sin.png"))
    _plot_spectrogram(sin, fs.get_hz(),
                     os.path.join(base_dir, "spec_narrowband_sin.tex"))

    _plot_spectrogram(brownian, fs.get_hz(),
                     os.path.join(base_dir, "spec_narrowband_brownian.png"))
    # _plot_spectrogram(brownian, fs.get_hz(),
    #                  os.path.join(base_dir, "spec_narrowband_brownian.tex"))

    _plot_spectrogram(brownian2, fs.get_hz(),
                     os.path.join(base_dir, "spec_narrowband_brownian2.png"))
    # _plot_spectrogram(brownian2, fs.get_hz(),
    #                  os.path.join(base_dir, "spec_narrowband_brownian2.tex"))

    _plot_spectrogram(am, fs.get_hz(),
                     os.path.join(base_dir, "spec_narrowband_am.png"))
    # _plot_spectrogram(am, fs.get_hz(),
    #                  os.path.join(base_dir, "spec_narrowband_am.tex"))

    _plot_spectrogram(fm, fs.get_hz(),
                     os.path.join(base_dir, "spec_narrowband_fm.png"))
    # _plot_spectrogram(fm, fs.get_hz(),
    #                  os.path.join(base_dir, "spec_narrowband_fm.tex"))

    _plot_spectrogram(chirp, fs.get_hz(),
                     os.path.join(base_dir, "spec_narrowband_chirp.png"))
    # _plot_spectrogram(chirp, fs.get_hz(),
    #                  os.path.join(base_dir, "spec_narrowband_chirp.tex"))

if __name__ == "__main__":
    main()
