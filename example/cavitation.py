import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import lps_sp.acoustical.broadband as lps_bb
import lps_synthesis.scenario.scenario as lps_scenario
import lps_utils.quantities as lps_qty

fs = lps_qty.Frequency.khz(16)
duration = lps_qty.Time.s(15)

ship = lps_scenario.ShipType.CONTAINERSHIP
speed = lps_qty.Speed.kt(5)
f_axis = lps_qty.Frequency.rpm(80)

n_blades = 4
n_shafts = 1
shaft_error = 5e-2
blade_error = 1e-2
blade_int_error = 0

n_samples = int(duration * fs)


t = np.linspace(0, duration.get_s(), n_samples, endpoint=False)

narrowband_total = np.zeros(n_samples)
narrowband_conv_total = np.zeros(n_samples)

A = 1.0        # Amplitude base
alpha = 0.01    # Fator de decaimento (1.5 é um valor típico)
narrowband = np.zeros_like(t)

n_harmonics = n_blades
for n in range(n_harmonics-1):
    amplitude = A * (2 - np.e**(alpha*n))
    frequency = (1 + n) * f_axis.get_hz() * n_blades
    fase = np.random.uniform(0, 2 * np.pi)  # Fase aleatória opcional
    narrowband += amplitude * np.cos(2 * np.pi * frequency * t + fase)

n_harmonics = 3
for n in range(n_harmonics):
    amplitude = 2 * A * (2 - np.e**(alpha*n))
    frequency = (1 + n) * f_axis.get_hz()
    fase = np.random.uniform(0, 2 * np.pi)  # Fase aleatória opcional
    narrowband += amplitude * np.cos(2 * np.pi * frequency * t + fase)

narrowband /= np.max(narrowband)

freqs, psd = ship.to_psd(fs=fs, speed=speed)
freqs_hz = [f.get_hz() for f in freqs]

broadband = lps_bb.generate(frequencies=np.array(freqs_hz),
                    psd_db=psd,
                    n_samples=n_samples,
                    fs=fs.get_hz())

mod_index = 0.8
signal = (1 + mod_index * narrowband) * broadband

def normalizar_sinal(sinal):
    sinal_max = np.max(np.abs(sinal))
    if sinal_max > 0:
        sinal = sinal / sinal_max
    return sinal

signal_norm = normalizar_sinal(signal)
signal_wav = np.int16(signal_norm * (2**15-1))

write("./result/cavitation.wav", int(fs.get_hz()), signal_wav)

# Visualizar os sinais
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, narrowband)
plt.title('Sinal de Banda Estreita (Conv dos Harmônicos)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

# Sinal de ruído
plt.subplot(2, 1, 2)
plt.plot(t, signal)
plt.title('Sinal Modulado (Banda Estreita + Ruído)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig("./result/cavitation.png")
