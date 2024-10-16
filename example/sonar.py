import scipy.io as scipy
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.environment.background as lps_bg
import lps_synthesis.scenario.sonar as lps_sonar
import lps_sp.acoustical.broadband as lps_bb

fs = lps_qty.Frequency.khz(16)
duration = lps_qty.Time.s(30)
n_samples=int(duration * fs)

bg = lps_bg.Background.random()
signal = bg.generate_bg_noise(fs=fs.get_hz(), n_samples=n_samples)

sensitivity = lps_qty.Sensitivity.db_v_p_upa(-165)
adc = lps_sonar.ADConverter(input_limits=(-5, 5), resolution=16)
sensor = lps_sonar.AcousticSensor(sensitivity=sensitivity, gain_db=80, adc=adc)


digital_signal = sensor.apply(signal)

scipy.wavfile.write("./result/background_noise.wav", int(fs.get_hz()), digital_signal)


freq, psd_signal = lps_bb.psd(signal=signal, fs=fs.get_hz())
freq2, psd_signal2 = lps_bb.psd(signal=digital_signal, fs=fs.get_hz(), db_unity=False)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # Primeiro subplot para a PSD
plt.semilogx(freq, psd_signal)
plt.title("Densidade Espectral de Potência (PSD)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Potência (dB)")
plt.grid(True)

plt.subplot(1, 2, 2)  # Segundo subplot para a PDF
plt.semilogx(freq2, psd_signal2)
plt.title("Densidade Espectral de Potência (PSD)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Potência (dB)")
plt.grid(True)

plt.tight_layout()
plt.savefig("./result/background_noise.png")