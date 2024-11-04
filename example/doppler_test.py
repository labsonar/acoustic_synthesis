import typing
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scipy

import lps_synthesis.propagation.models as lps_model
import lps_utils.quantities as lps_qty



def apply_doppler(input_data: np.array, speeds: np.array, sound_speed: float):

    num_blocks = len(speeds)
    samples_per_block = len(input_data)//num_blocks

    output = np.array([])
    for i_block in range(num_blocks):
        doppler_factor = (sound_speed + speeds[i_block]) / sound_speed

        scaled_data = scipy.resample(
                input_data[i_block * samples_per_block:(i_block+1) * samples_per_block - 1],
                int(samples_per_block//doppler_factor))

        output = np.concatenate((output, scaled_data))

    return output



sound_speed = lps_qty.Speed.m_s(1500)  # velocidade do som na água (m/s)
sampling_rate = 16000  # taxa de amostragem do hidrofone (Hz)
duration = 5
n_samples = duration * sampling_rate

ship1_velocity_range = [lps_qty.Speed.kt(s) for s in np.linspace(-10, 10, 100)]  # velocidades do navio 1
ship2_velocity_range = [lps_qty.Speed.kt(s) for s in np.linspace(-50, 50, 100)]  # velocidades do navio 2
freq_ship1 = 6000  # frequência senoidal do navio 1 (Hz)
freq_ship2 = 2000  # frequência senoidal do navio 2 (Hz)

time = np.linspace(0, duration, n_samples, endpoint=False)


noise_ship1 = np.random.normal(0, 0.5, n_samples) + 2 * np.sin(2 * np.pi * freq_ship1 * time)
noise_ship2 = np.random.normal(0, 0.5, n_samples) + 2 * np.sin(2 * np.pi * freq_ship2 * time)


signal_ship1 = lps_model.apply_doppler(noise_ship1, ship1_velocity_range, sound_speed)
signal_ship2 = lps_model.apply_doppler(noise_ship2, ship2_velocity_range, sound_speed)

final_n_samples = np.min((len(signal_ship1), len(signal_ship2)))
hydrophone_signal = signal_ship1[:final_n_samples] + signal_ship2[:final_n_samples]

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