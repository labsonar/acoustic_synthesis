""" Simple Ship Noise test. """
import csv
import os

import numpy as np
import scipy.io as scipy
import matplotlib.pyplot as plt
import tikzplotlib

import lps_sp.acoustical.broadband as lps_bb
import lps_synthesis.scenario.scenario as lps_scenario
import lps_utils.quantities as lps_qty

output_dir = "./plot"
os.makedirs(output_dir, exist_ok=True)

fs = lps_qty.Frequency.khz(16)
duration = lps_qty.Time.s(300)
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

    plt.semilogx(freqs_hz, psd, label=str(ship).capitalize().replace("_", " "))

plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB ref 1 µPa @ 1m / √Hz)")
plt.savefig(os.path.join(output_dir, "ships_psd.png"), bbox_inches='tight')
tikzplotlib.save(os.path.join(output_dir, "ships_psd.tex"))
plt.close()


plt.figure(figsize=(12, 6))
for ship in lps_scenario.ShipType:

    freqs, psd = ship.to_psd(fs=fs, speed=speed)
    freqs_hz = [f.get_hz() for f in freqs]

    if not csv_data:
        csv_data = [[freq] for freq in freqs_hz]

    for i, value in enumerate(psd):
        csv_data[i].append(value)
    header.append(str(ship))

    audio_signal = lps_bb.generate(frequencies=np.array(freqs_hz),
                                   psd_db=psd,
                                   n_samples=n_samples,
                                   fs=fs.get_hz())

    est_freqs, est_psd_db = lps_bb.psd(signal=audio_signal, fs=fs.get_hz(), window_size=1024*2, overlap=0.9)
    est_freqs = est_freqs[:-1]
    est_psd_db = est_psd_db[:-1]

    num_freqs = len(est_freqs)
    log_freqs = np.logspace(np.log10(est_freqs.min()), np.log10(est_freqs.max()), num=500)

    log_psd_db = np.interp(log_freqs, est_freqs, est_psd_db)
    plt.semilogx(log_freqs, log_psd_db, label=str(ship).capitalize().replace("_", " "))

plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB ref 1 µPa @ 1m / √Hz)")
plt.savefig(os.path.join(output_dir, "ships_psd_simulated.png"), bbox_inches='tight')
tikzplotlib.save(os.path.join(output_dir, "ships_psd_simulated.tex"))
plt.close()
