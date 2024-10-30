""" Example of use of propagation module
"""
import time
import numpy as np

import matplotlib.pyplot as plt
import scipy.signal as scipy
import scipy.io.wavfile as wavfile

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.channel as lps_channel

sample_frequency = lps_qty.Frequency.khz(16)

lps_channel.TEMP_DEFAULT_DIR = "./result/propagation"


start_time = time.time()
channel = lps_channel.PredefinedChannel.BASIC.get_channel()
end_time = time.time()
print("Estimate channel: ", end_time-start_time)

channel.get_ir().print_h_t_tau('./result/ir.png')

Ts = 1/sample_frequency.get_hz()
s_n_samples = int(sample_frequency.get_hz() * 10)
s_t = np.arange(0, s_n_samples * Ts, Ts)

r_n_samples = s_n_samples
r_t = [ lps_qty.Distance.m(((i/r_n_samples) - 0.5) * 800) for i in range(r_n_samples)]

input_signal = np.random.randn(s_n_samples) * 0.8

cutoff_freq = lps_qty.Frequency.hz(1000)
b, a = scipy.butter(1, cutoff_freq / (sample_frequency / 2), btype='low')

input_signal = scipy.filtfilt(b, a, input_signal)

# s_freqs = [3000, 5500]
# for freq in s_freqs:
#     input_signal += np.sin(2 * np.pi * freq * s_t) * 0.2

start_time = time.time()
final_signal = channel.propagate(input_signal, lps_qty.Distance.m(10), r_t)
end_time = time.time()
print("Propagate signal: ", end_time-start_time)


f, t, S = scipy.spectrogram(final_signal, fs=sample_frequency.get_hz(), nperseg=2048)

plt.figure(figsize=(10, 6))
plt.imshow(20 * np.log10(np.abs(S)), interpolation='none', aspect='auto')
plt.title('Espectrograma do Sinal')
plt.ylabel('Frequência [Hz]')
plt.xlabel('Tempo [s]')
plt.colorbar(label='Intensidade [dB]')
plt.savefig("./result/propagation.png")
plt.close()


final_signal_normalized = final_signal / np.max(np.abs(final_signal))
sample_rate = int(sample_frequency.get_hz())
wavfile.write('./result/propagation.wav', sample_rate, final_signal_normalized.astype(np.float32))
