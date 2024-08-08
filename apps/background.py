"""
Background script to generate a sample of noise in some enviroment condition
"""
import argparse
import os
import datetime

import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt


import lps_sp.signal as lps_signal
import lps_sp.acoustical.broadband as lps_bb
import lps_synthesis.background as lps

def main(background: lps.Background, fs: int, duration: float):
    """ Script to generate a acoustical enviroment noise

    Args:
        background (lps.Background): Background class
        fs (int): sample frequency
        duration (float): duration in seconds
    """    """"""

    base_dir = "./results"
    os.makedirs(base_dir, exist_ok = True)

    n_samples = int(fs * duration)
    noise = background.generate_bg_noise(n_samples=n_samples)

    filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    wav_filename = os.path.join(base_dir, f'{filename}.wav')
    psd_filename = os.path.join(base_dir, f'{filename}.png')

    norm = lps_signal.Normalization.MIN_MAX_ZERO_CENTERED
    scipy_wav.write(wav_filename, fs, norm.apply(noise))

    print(f"Noise saved as {wav_filename}")

    frequencies, psd = lps_bb.psd(noise, fs)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, psd, label=str(background))

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB re 1μPa^2/Hz)')
    plt.semilogx()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(psd_filename)


if __name__ == '__main__':

    bg = lps.Background.random()

    parser = argparse.ArgumentParser(description="Generate a background noise WAV file.")
    parser.add_argument('-r','--rain_value', type=float, default=None,
                            help='Rain intensity value between 0 and 4.')
    parser.add_argument('-s','--sea_value', type=float, default=None,
                            help='Sea state value between 0 and 6.')
    parser.add_argument('-S','--shipping_value', type=float, default=None,
                            help='Shipping noise value between 0 and 7.')
    parser.add_argument('--fs', type=int, default=48000,
                            help='Sampling frequency in Hz.')
    parser.add_argument('--seconds', type=float, default=5,
                            help='Duration of the generated noise in seconds.')

    args = parser.parse_args()

    if args.rain_value is not None:
        bg.rain_value = args.rain_value
    if args.sea_value is not None:
        bg.sea_value = args.sea_value
    if args.shipping_value is not None:
        bg.shipping_value = args.shipping_value

    main(bg, args.fs, args.seconds)
