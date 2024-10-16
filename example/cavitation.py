import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import lps_sp.acoustical.broadband as lps_bb
import lps_synthesis.scenario.scenario as lps_scenario
import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.acoustical_channel as lps_channel

fs = lps_qty.Frequency.khz(16)
duration = lps_qty.Time.s(15)

ship = lps_scenario.ShipType.CONTAINERSHIP
speed = lps_qty.Speed.kt(5)
f_axis = lps_qty.Frequency.rpm(80)

n_blades = 4
n_shafts = 2
shaft_error = 5e-2
blade_error = 1e-3

n_samples = int(duration * fs)


t = np.linspace(0, duration.get_s(), n_samples, endpoint=False)

narrowband_total = np.zeros(n_samples)
for eixo in range(n_shafts):

    eixo_variation = f_axis.get_hz() * (1 + shaft_error * np.random.randn())
    f_RPM1_vec = np.ones(n_samples) * eixo_variation

    f_21 = np.cumsum(f_RPM1_vec) / fs.get_hz()

    m21 = np.zeros((n_blades, n_samples))
    for har in range(1, n_blades + 1):
        blade_phase_shift = (har - 1) * 2 * np.pi / n_blades
        harmonic_variation = 1 + blade_error * np.random.randn()
        m21[har - 1, :] = np.cos(2 * np.pi * f_21 * har * harmonic_variation + blade_phase_shift)

    narrowband_eixo = np.sum(m21, axis=0)
    narrowband_total += narrowband_eixo

narrowband_total /= n_shafts * n_blades


freqs, psd = ship.to_psd(fs=fs, speed=speed)
freqs_hz = [f.get_hz() for f in freqs]

broadband = lps_bb.generate(frequencies=np.array(freqs_hz),
                    psd_db=psd,
                    n_samples=n_samples,
                    fs=fs.get_hz())
# broadband = 10 * np.random.randn(n_samples)

mod_index = 1
signal = (1 + mod_index * narrowband_total) * broadband

def normalizar_sinal(sinal):
    sinal_max = np.max(np.abs(sinal))
    if sinal_max > 0:
        sinal = sinal / sinal_max
    return sinal

broadband_norm = normalizar_sinal(broadband)
broadband_wav = np.int16(broadband_norm * (2**15-1))

signal_norm = normalizar_sinal(signal)
signal_wav = np.int16(signal_norm * (2**15-1))


source_depths = [lps_qty.Distance.m(5),
                 lps_qty.Distance.m(6),
                 lps_qty.Distance.m(10),
                 lps_qty.Distance.m(15)]

desc = lps_channel.Description()
desc.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))
desc.add(lps_qty.Distance.m(50), lps_layer.BottomType.CHALK)

channel = lps_channel.Channel(
                description = desc,
                source_depths = source_depths,
                sensor_depth = lps_qty.Distance.m(40),
                max_distance = lps_qty.Distance.m(1000),
                max_distance_points = 200,
                sample_frequency = fs,
                temp_dir = './result/propagation')


r_n_samples = n_samples
r_t = [ lps_qty.Distance.m((i/r_n_samples - 0.5) * 150) for i in range(r_n_samples)]

prop_signal = channel.propagate(signal, source_depths[1], r_t)

prop_signal_norm = normalizar_sinal(prop_signal)
prop_signal_wav = np.int16(prop_signal_norm * (2**15-1))


write("./result/ship_noise.wav", int(fs.get_hz()), broadband_wav)
write("./result/modulated_ship_noise.wav", int(fs.get_hz()), signal_wav)
write("./result/propagated_ship_noise.wav", int(fs.get_hz()), prop_signal_wav)

# Visualizar os sinais
plt.figure(figsize=(12, 8))

# Banda estreita (soma dos harmônicos)
plt.subplot(4, 1, 1)
plt.plot(t, narrowband_total)
plt.title('Sinal de Banda Estreita (Soma dos Harmônicos)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

# Sinal de ruído
plt.subplot(4, 1, 2)
plt.plot(t, broadband)
plt.title('Ruído de Banda Larga (Gaussiano)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

# Sinal modulado
plt.subplot(4, 1, 3)
plt.plot(t, signal)
plt.title('Sinal Modulado (Banda Estreita + Ruído)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

# Sinal modulado
plt.subplot(4, 1, 4)
plt.plot(t, prop_signal)
plt.title('Sinal Propagado (Banda Estreita + Ruído + Canal)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig("./result/cavitation.png")

print("Arquivos .wav salvos: 'ruido_original.wav' e 'ruido_modulado_banda_estreita.wav'")
