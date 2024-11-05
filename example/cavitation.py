import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import lps_sp.acoustical.broadband as lps_bb
import lps_synthesis.scenario.scenario as lps_scenario
import lps_utils.quantities as lps_qty

fs = lps_qty.Frequency.khz(16)
duration = lps_qty.Time.s(5)

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

# A = 1.0        # Amplitude base
alpha = 0.01    # Fator de decaimento (1.5 é um valor típico)


def generate_harmonic_intensities(n_harmonics, k, alpha=0.5, A=1.0, G=1.5, noise_std=0.1):
    intensities = []
    rn = np.random.normal(0, noise_std, k)
    for n in range(1, n_harmonics + 1):
        
        # Reforço para múltiplos de k
        if n % k == 0:
            base_intensity = G * A / ((n/k) ** alpha)
        else:
            base_intensity = A / (n ** alpha)
        
        # Adiciona ruído leve
        intensity = base_intensity + rn[n%k]
        intensities.append(intensity)
    
    return np.array(intensities)

alpha = 2
# A = np.array([0.108666, 0.051330, 0.051330, 0.179160, 0.051330, 0.051330])
A = generate_harmonic_intensities(n_harmonics=20, k=n_blades)

# for _ in range(3):
#     plt.plot(generate_harmonic_intensities(n_harmonics=20, k=4), 'o')

A0 =  np.sum(A)/alpha

total_energy = A0**2 + np.sum(A**2)/2

A0 /= np.sqrt(total_energy)
A /= np.sqrt(total_energy)

print(A0, " ", A)
plt.plot(A, 'o')
plt.savefig("./result/cavitation_index.png")

narrowband = np.ones_like(t) * A0

n_harmonics = len(A)
f_ref = f_axis.get_hz()
for n in range(n_harmonics):
    # amplitude = A * (2 - np.e**(alpha*n))
    amplitude = A[n]
    frequency = (1 + n) * f_ref
    fase = 0# np.random.uniform(0, 2 * np.pi)  # Fase aleatória opcional
    narrowband += amplitude * np.cos(2 * np.pi * frequency * t + fase)

# f_ref = f_axis.get_hz() * (1 + 0.05)
# for n in range(n_harmonics):
#     # amplitude = A * (2 - np.e**(alpha*n))
#     amplitude = [0.8, 0.3, 0.4, 1][n]
#     frequency = (1 + n) * f_ref
#     fase = 0# np.random.uniform(0, 2 * np.pi)  # Fase aleatória opcional
#     narrowband2 += amplitude * np.cos(2 * np.pi * frequency * t + fase)

# # n_harmonics = 3
# # for n in range(n_harmonics):
# #     amplitude = 2 * A * (2 - np.e**(alpha*n))
# #     frequency = (1 + n) * f_axis.get_hz()
# #     fase = np.random.uniform(0, 2 * np.pi)  # Fase aleatória opcional
# #     narrowband += amplitude * np.cos(2 * np.pi * frequency * t + fase)

# narrowband /= np.max(narrowband)
# narrowband2 /= np.max(narrowband2)

freqs, psd = ship.to_psd(fs=fs, speed=speed)
freqs_hz = [f.get_hz() for f in freqs]

broadband = lps_bb.generate(frequencies=np.array(freqs_hz),
                    psd_db=psd,
                    n_samples=n_samples,
                    fs=fs.get_hz())

# mod_index = 0.5
signal = (narrowband) * broadband
# signal2 = (1 + mod_index * narrowband2) * broadband

# signal = signal + signal2

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

# plt.tight_layout()
plt.savefig("./result/cavitation.png")
