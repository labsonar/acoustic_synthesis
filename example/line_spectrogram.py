import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram


sample_rate, data = wavfile.read("./result/scenario.wav")
# sample_rate, data = wavfile.read("./result/data/recreational_4m_trafego_de_fundo/scenario.wav")
# sample_rate, data = wavfile.read("./result/data/planar/scenario.wav")

if data.ndim > 1:
    data = data[:,0]

# Gerar o espectrograma com resolução de 1 Hz
frequencies, times, Sxx = spectrogram(data, fs=sample_rate, nperseg=sample_rate * 2, noverlap=sample_rate - 2048)

# Selecionar a faixa de frequência desejada entre 3950 Hz e 4050 Hz
freq_min = 3970
freq_max = 4030
freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)

# Plotar o espectrograma na faixa selecionada
plt.pcolormesh(times, frequencies[freq_mask], 20 * np.log10(Sxx[freq_mask, :]), shading='gouraud')
plt.ylabel("Frequência (Hz)")
plt.xlabel("Tempo (s)")
plt.title("Espectrograma de 3950 Hz a 4050 Hz")
plt.colorbar(label="Intensidade (dB)")
plt.savefig("./result/line_spectrogram.png")
