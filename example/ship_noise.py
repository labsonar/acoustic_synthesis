""" Simple Ship Noise test. """
import csv

import numpy as np
import scipy.io as scipy
import matplotlib.pyplot as plt

import lps_sp.acoustical.broadband as lps_bb
import lps_synthesis.scenario.scenario as lps_scenario
import lps_utils.quantities as lps_qty

fs = lps_qty.Frequency.khz(16)
duration = lps_qty.Time.s(5)
speed = lps_qty.Speed.kt(5)
n_samples = int(duration * fs)

all_audio = []
csv_data = []
header = ['Frequency (Hz)']

plt.figure(figsize=(12, 6))
for ship in lps_scenario.ShipType:

    freqs, psd = ship.to_psd(fs=fs, speed=speed)
    freqs_hz = [f.get_hz() for f in freqs]

    if not csv_data:
        csv_data = [[freq] for freq in freqs_hz]

    for i, value in enumerate(psd):
        csv_data[i].append(value)
    header.append(str(ship))

    plt.semilogx(freqs_hz, psd, label=str(ship))

    audio_signal = lps_bb.generate(frequencies=np.array(freqs_hz),
                                   psd_db=psd,
                                   n_samples=n_samples,
                                   fs=fs.get_hz())

    est_freqs, est_psd_db = lps_bb.psd(signal=audio_signal, fs=fs.get_hz(), window_size=2048)
    # plt.semilogx(est_freqs, est_psd_db, label=str(ship))

    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    audio_signal_int16 = np.int16(audio_signal * (2**15-1))
    all_audio.append(audio_signal_int16)

plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig("./result/all_ships_psd.png", bbox_inches='tight')
plt.close()

with open('./result/frequency_psd_data.csv', mode='w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(csv_data)

all_audio_concat = np.concatenate(all_audio)
scipy.wavfile.write('./result/all_ships_audio.wav', int(fs.get_hz()), all_audio_concat)
