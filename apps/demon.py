import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tikz

from scipy.io.wavfile import read
from librosa import fft_frequencies, frames_to_time, stft
from sympy import factorint
from scipy.signal import cheb2ord, cheby2, decimate, hilbert, lfilter

import lps_sp.signal as lps_signal


def _get_demon_steps(fs_in, fs_out=50):
    if (fs_in % fs_out) != 0:
        raise ValueError("fs_in não divisível por fs_out")

    factors = factorint(int(fs_in / fs_out))
    factor_list = [f for factor, count in factors.items() for f in [factor] * count]

    decimate_ratio1 = 1
    decimate_ratio2 = 1
    add_one = True

    while factor_list:
        if len(factor_list) == 1:
            part1 = factor_list.pop()
            part2 = 1
        elif len(factor_list) == 2:
            part1 = factor_list.pop(0)
            part2 = factor_list.pop()
        else:
            part1 = factor_list.pop(0) * factor_list.pop()
            part2 = 1

        if add_one:
            decimate_ratio1 *= part1
            decimate_ratio2 *= part2
        else:
            decimate_ratio1 *= part2
            decimate_ratio2 *= part1
        add_one = not add_one

    return [decimate_ratio1, decimate_ratio2]

def _demon(data, fs, n_fft=512, max_freq=50, overlap_ratio=0.25, apply_bandpass=True,
          bandpass_specs=None, method='abs'):

    [decimate_ratio1, decimate_ratio2] = _get_demon_steps(fs, max_freq)
    x = data.copy()

    nyq = fs / 2
    if apply_bandpass:
        if bandpass_specs is None:
            wp = [1000 / nyq, 2000 / nyq]
            ws = [700 / nyq, 2300 / nyq]
            rp = 0.5
            As = 50
        else:
            fp = bandpass_specs["fp"]
            fs_band = bandpass_specs["fs"]
            wp = np.array(fp) / nyq
            ws = np.array(fs_band) / nyq
            rp = bandpass_specs["rs"]
            As = bandpass_specs["as"]

        N, wc = cheb2ord(wp, ws, rp, As)
        b, a = cheby2(N, rs=As, Wn=wc, btype='bandpass', output='ba', analog=True)
        x = lfilter(b, a, x, axis=0)

    if method == 'hilbert':
        x = hilbert(x)
    elif method == 'abs':
        x = np.abs(x)
    else:
        raise ValueError("Método inválido")

    x = decimate(x, decimate_ratio1, ftype='fir', zero_phase=False)
    x = decimate(x, decimate_ratio2, ftype='fir', zero_phase=False)

    final_fs = (fs // decimate_ratio1) // decimate_ratio2
    x = x / np.max(np.abs(x))
    x = x - np.mean(x)

    fft_over = math.floor(n_fft - 2 * max_freq * overlap_ratio)
    sxx = stft(x, window='hann', win_length=n_fft, hop_length=n_fft - fft_over, n_fft=n_fft)
    freq = fft_frequencies(sr=final_fs, n_fft=n_fft)
    time = frames_to_time(np.arange(sxx.shape[1]), sr=final_fs, hop_length=(n_fft - fft_over))

    sxx = np.abs(sxx)
    sxx, freq = sxx[8:, :], freq[8:]

    return np.transpose(sxx), freq, time

def main(wav_path: str, save_mode: str) -> None:

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"File not found: {wav_path}")

    output_dir = os.path.dirname(os.path.abspath(wav_path))
    output_base = os.path.splitext(os.path.basename(wav_path))[0]

    fs, signal = read(wav_path)
    S, f, t = _demon(signal, fs)

    output_path = os.path.join(output_dir, f"{output_base}_demon")

    plt.figure()
    if len(signal) / fs < 10:
        plt.plot(f * 60, np.mean(S, axis=0))
        plt.ylabel('Amplitude')
    else:
        plt.imshow(S, aspect='auto', extent=[f[0] * 60, f[-1] * 60, t[-1], t[0]])
        plt.ylabel('Time')

    plt.title('DEMON')
    plt.xlabel('Frequency [rpm]')
    plt.tight_layout()

    if save_mode in ("wav", "both"):
        plt.savefig(output_path + ".png")
    if save_mode in ("tex", "both"):
        tikz.save(output_path + ".tex")

    print(f"File saved at: {output_path}[.png/.tex]")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run DEMON analysis on a .wav file.")
    parser.add_argument("wav_path", type=str, help="Path to the .wav file to analyze")
    parser.add_argument("--save", choices=["wav", "tex", "both"], default="wav",
                        help="Output format to save (default: wav)")

    args = parser.parse_args()
    main(args.wav_path, args.save)
