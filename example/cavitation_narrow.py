import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Bibliotecas da LPS Soundscape
import lps_sp.acoustical.broadband as lps_bb
import lps_synthesis.scenario.scenario as lps_scenario
import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.acoustical_channel as lps_channel

fs = lps_qty.Frequency.khz(16)
duration = lps_qty.Time.s(5)

ship = lps_scenario.ShipType.CONTAINERSHIP
speed = lps_qty.Speed.kt(5)
f_axis = lps_qty.Frequency.rpm(90)
n_blades = 4
n_shafts = 1
shaft_error = 0
blade_error = 0

n_samples = int(duration * fs)


t = np.linspace(0, duration.get_s(), n_samples, endpoint=False)


narrowband_total = np.zeros(n_samples)


for eixo in range(n_shafts):

    eixo_variation = f_axis.get_hz() * (1 + shaft_error * np.random.randn())
    f_RPM1_vec = np.ones(n_samples) * eixo_variation

    f_21 = np.cumsum(f_RPM1_vec) / fs.get_hz()

    m21 = np.zeros((n_blades, n_samples))
    for har in range(1, n_blades + 1):
        harmonic_variation = 1 + blade_error * np.random.randn()
        m21[har - 1, :] = np.cos(2 * np.pi * f_21 * har * harmonic_variation)

    narrowband_eixo = np.sum(m21, axis=0)
    narrowband_total += narrowband_eixo


narrowband_total /= n_shafts * n_blades


plt.figure(figsize=(12, 8))
plt.plot(t, narrowband_total)
plt.title(f'Sinal de Banda Estreita (Soma dos Harmônicos) para {n_shafts} Eixos com Imperfeições')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig("./result/cavitation_narrow.png")
