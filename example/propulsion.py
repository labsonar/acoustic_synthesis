""" Simple propulsion test. """
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import lps_sp.acoustical.broadband as lps_bb
import lps_synthesis.scenario.scenario as lps_scenario
import lps_utils.quantities as lps_qty

fs = lps_qty.Frequency.khz(16)
duration = lps_qty.Time.s(15)

n_samples = int(duration * fs)
t = np.linspace(0, duration.get_s(), n_samples, endpoint=False)

ship = lps_scenario.ShipType.CONTAINERSHIP
speed = lps_qty.Speed.kt(5)
f_axis = lps_qty.Frequency.rpm(80)


freqs, psd = ship.to_psd(fs=fs, speed=speed)
freqs_hz = [f.get_hz() for f in freqs]

broadband = lps_bb.generate(frequencies=np.array(freqs_hz),
                    psd_db=psd,
                    n_samples=n_samples,
                    fs=fs.get_hz())

prop = lps_scenario.CavitationNoise(ship_type=ship,
                               n_blades = 4,
                               n_shafts = 1,
                               shaft_error = 0,
                               blade_error = 0)

_, max_speed = ship.get_speed_range()
min_speed = lps_qty.Speed.kt(4)
speed_samples = 50
speeds = [min_speed + (max_speed-min_speed) * (i/speed_samples) for i in range(speed_samples)]
# speeds = [max_speed for i in range(speed_samples)]

signal, narrowband_total = prop.modulate_noise(broadband=broadband, speeds=speeds, fs=fs)


def normalizar_sinal(sinal):
    """ normalize signal by max (-1,+1)"""
    sinal_max = np.max(np.abs(sinal))
    if sinal_max > 0:
        sinal = sinal / sinal_max
    return sinal

broadband_norm = normalizar_sinal(broadband)
broadband_wav = np.int16(broadband_norm * (2**15-1))

signal_norm = normalizar_sinal(signal)
signal_wav = np.int16(signal_norm * (2**15-1))

write("./result/ship_noise.wav", int(fs.get_hz()), broadband_wav)
write("./result/modulated_ship_noise.wav", int(fs.get_hz()), signal_wav)


plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, narrowband_total)
plt.title('Sinal de Banda Estreita (Soma dos Harmônicos)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(t, broadband)
plt.title('Ruído de Banda Larga (Gaussiano)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(t, signal)
plt.title('Sinal Modulado (Banda Estreita + Ruído)')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig("./result/cavitation.png")

print("Arquivos .wav salvos: 'ruido_original.wav' e 'ruido_modulado_banda_estreita.wav'")
