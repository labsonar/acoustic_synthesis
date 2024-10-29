import math
import numpy as np

import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from librosa import fft_frequencies, frames_to_time, stft
from sympy import factorint
from scipy.signal import cheb2ord, cheby2, convolve, decimate, hilbert, lfilter, spectrogram

from lps_sp.signal import tpsw

def get_demon_steps(fs_in, fs_out=50):
    if (fs_in % fs_out) != 0:
        raise ValueError("fs_in não divisivel por fs_out")

    factors = factorint(int(fs_in / fs_out))

    factor_list = []
    for factor, expo in factors.items():
        factor_list += [factor] * expo

    decimate_ratio1 = 1
    decimate_ratio2 = 1

    add_one = True

    while len(factor_list) > 0:
        part1 = 1
        part2 = 1

        if len(factor_list) == 1:
                part1 = factor_list.pop(0)
        elif len(factor_list) == 2:
                if decimate_ratio1 == decimate_ratio2:
                    part1 = factor_list.pop(0)
                    part2 = factor_list.pop(-1)
                else:
                    part1 = factor_list.pop(0) * factor_list.pop(-1)
        else:
                part1 = factor_list.pop(0) * factor_list.pop(-1)

        if add_one:
            decimate_ratio1 = decimate_ratio1 * part1
            decimate_ratio2 = decimate_ratio2 * part2
        else:
            decimate_ratio1 = decimate_ratio1 * part2
            decimate_ratio2 = decimate_ratio2 * part1
        
        add_one = not add_one

    return [decimate_ratio1, decimate_ratio2]

def demon(data, fs, n_fft=512, max_freq=50, overlap_ratio=0.25, apply_bandpass=True, bandpass_specs=None, method='abs'):

    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be of type numpy.ndarray. %s was passed" % type(data))

    [decimate_ratio1, decimate_ratio2] = get_demon_steps(fs, max_freq)

    x = data.copy()

    # first_pass_sr = 1250  # 31250/25
    # q1 = round(fs/first_pass_sr)  # 25 for 31250 sample rate ; decimatio ratio for 1st pass
    # q2 = round((fs/q1)/(2*max_freq))  # decimatio ratio for 2nd pass

    fft_over = math.floor(n_fft-2*max_freq*overlap_ratio)
    nyq = fs/2

    if apply_bandpass:
        if bandpass_specs is None:
            wp = [1000/nyq, 2000/nyq]
            ws = [700/nyq, 2300/nyq]
            rp = 0.5
            As = 50
        elif isinstance(bandpass_specs, dict):
            try:
                fp = bandpass_specs["fp"]
                fs = bandpass_specs["fs"]

                wp = np.array(fp)/nyq
                ws = np.array(fs)/nyq

                rp = bandpass_specs["rs"]
                As = bandpass_specs["as"]
            except KeyError as e:
                raise KeyError("Missing %s specification for bandpass filter" % e)
        else:
            raise ValueError("bandpass_specs must be of type dict. %s was passed" % type(bandpass_specs))

        N, wc = cheb2ord(wp, ws, rp, As)
        b, a = cheby2(N, rs=As, Wn=wc, btype='bandpass', output='ba', analog=True)
        x = lfilter(b, a, x, axis=0)

    if method == 'hilbert':
        x = hilbert(x)
    elif method == 'abs':
        x = np.abs(x)  # demodulation
    else:
        raise ValueError("Method not found")

    x = decimate(x, decimate_ratio1, ftype='fir', zero_phase=False)
    x = decimate(x, decimate_ratio2, ftype='fir', zero_phase=False)

    final_fs = (fs//decimate_ratio1)//decimate_ratio2

    x /= x.max()
    x -= np.mean(x)
    sxx = stft(x,
               window=('hann'),
               win_length=n_fft,
               hop_length=(n_fft - fft_over),
               n_fft=n_fft)
    freq = fft_frequencies(sr=final_fs, n_fft=n_fft)
    time = frames_to_time(np.arange(0, sxx.shape[1]),
                          sr=final_fs, hop_length=(n_fft - fft_over))

    sxx = np.absolute(sxx)
    sxx[sxx == 0] = 1e-9
    sxx = 20*np.log10(sxx)
    sxx = sxx - tpsw(sxx)
    sxx[sxx < -0.2] = 0

    sxx, freq = sxx[8:, :], freq[8:]  # ??

    return np.transpose(sxx), freq, time

# fs, signal = read("./result/modulated_ship_noise.wav")
fs, signal = read("./result/cavitation.wav")
# fs, signal = read("./result/sum.wav")

S, f, t = demon(signal, fs)

if len(signal)/fs < 10:
    plt.plot(f*60, np.mean(S,axis=0))
    plt.ylabel('Amplitude')

else:
    plt.imshow(S, aspect='auto', extent=[
                                        f[0] * 60,
                                        f[-1] * 60,
                                        t[0],
                                        t[-1]]
                        )
    plt.ylabel('Time')

plt.title('Demon')
plt.xlabel('Frequency [rpm]')
plt.savefig("./result/demon.png")