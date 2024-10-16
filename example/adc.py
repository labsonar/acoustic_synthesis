import numpy as np
import matplotlib.pyplot as plt

import lps_synthesis.scenario.scenario as lps_scenario


frequency = 200
amplitude = 5
sample_rate = 16000
duration = 0.01

t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal = amplitude * np.sin(2 * np.pi * frequency * t)

resolutions = [8, 12, 16, 32]
plt.figure(figsize=(12, 8))

plt.subplot(len(resolutions) + 1, 1, 1)
plt.plot(t, signal, label="Sinal Original (Volts)", color='blue', alpha=0.5)
plt.title("Original")
plt.xlabel("Tempo (s)")
plt.ylabel("Valor convertido")

print("Sinal: ", np.max(signal))

for i, res in enumerate(resolutions):
    adc = lps_scenario.ADConverter(input_limits=(0, 3), resolution=res)
    converted_signal = adc.convert(signal)

    print(res, " bits: ", np.max(converted_signal), "/", 2**(res - 1) - 1)

    plt.subplot(len(resolutions) + 1, 1, i+2)
    plt.step(t, converted_signal, label=f"Convertido com {res} bits", color='red', where='mid')
    plt.title(f"Conversão A/D com {res} bits")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Valor convertido")

plt.tight_layout()
plt.savefig("./result/adc.png")
