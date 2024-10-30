import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scipy

# Constantes
sound_speed = 1500  # velocidade do som na água (m/s)
sampling_rate = 16000  # taxa de amostragem do hidrofone (Hz)
block_duration = 0.05  # duração de cada bloco de simulação (segundos)
num_blocks = 100  # número de blocos
num_samples_per_block = int(block_duration * sampling_rate)

# Parâmetros dos navios (faixas de velocidade e frequências)
ship1_velocity_range = np.linspace(-10, 10, num_blocks)  # velocidades do navio 1
ship2_velocity_range = np.linspace(-50, 50, num_blocks)  # velocidades do navio 2
freq_ship1 = 6000  # frequência senoidal do navio 1 (Hz)
freq_ship2 = 2000  # frequência senoidal do navio 2 (Hz)

# Vetores para armazenar os sinais concatenados dos navios
signal_ship1_concat = np.array([])
signal_ship2_concat = np.array([])

# Loop para gerar cada bloco de sinal
for i in range(num_blocks):
    # Velocidades dos navios no bloco atual
    ship1_velocity = ship1_velocity_range[i]
    ship2_velocity = ship2_velocity_range[i]

    # Fator Doppler para cada navio
    doppler_factor_ship1 = (sound_speed + ship1_velocity) / sound_speed
    doppler_factor_ship2 = (sound_speed + ship2_velocity) / sound_speed

    # Gera o tempo comprimido/expandido para cada navio devido ao efeito Doppler
    num_samples_doppler_ship1 = int(block_duration * sampling_rate * doppler_factor_ship1)
    num_samples_doppler_ship2 = int(block_duration * sampling_rate * doppler_factor_ship2)
    time_doppler_ship1 = np.linspace(0, block_duration * doppler_factor_ship1, num_samples_doppler_ship1, endpoint=False)
    time_doppler_ship2 = np.linspace(0, block_duration * doppler_factor_ship2, num_samples_doppler_ship2, endpoint=False)

    # Gera o sinal de ruído com componente senoidal para cada navio no domínio Doppler
    noise_ship1 = np.random.normal(0, 0.5, num_samples_doppler_ship1) + 2 * np.sin(2 * np.pi * freq_ship1 * time_doppler_ship1)
    noise_ship2 = np.random.normal(0, 0.5, num_samples_doppler_ship2) + 2 * np.sin(2 * np.pi * freq_ship2 * time_doppler_ship2)

    # Re-amostra os sinais para a taxa de 16 kHz, para alinhar ao tempo do hidrofone
    signal_ship1 = scipy.resample(noise_ship1, num_samples_per_block)
    signal_ship2 = scipy.resample(noise_ship2, num_samples_per_block)

    # Concatena os blocos no vetor final
    signal_ship1_concat = np.concatenate((signal_ship1_concat, signal_ship1))
    signal_ship2_concat = np.concatenate((signal_ship2_concat, signal_ship2))

# Sinal combinado no hidrofone
hydrophone_signal = signal_ship1_concat + signal_ship2_concat

# Espectrograma do sinal combinado
frequencies, times, Sxx = scipy.spectrogram(hydrophone_signal, fs=sampling_rate, nperseg=1024)

# Plot dos sinais e espectrograma
plt.figure(figsize=(12, 10))

plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.colorbar(label="Intensidade (dB)")
plt.ylabel("Frequência (Hz)")
plt.xlabel("Tempo (s)")
plt.title("Espectrograma do Sinal Recebido no Hidrofone")

plt.tight_layout()
plt.savefig("./result/doppler_test.png")
plt.close()
