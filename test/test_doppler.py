import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_signal
import lps_synthesis.propagation.channel_response as lps_propag_rsp

import numpy as np
from scipy.signal import resample_poly, windows

def main():

    output_dir = "./result"
    os.makedirs(output_dir, exist_ok=True)

    fs = lps_qty.Frequency.khz(16)
    duration = lps_qty.Time.s(10)
    t_offset_plot = lps_qty.Time.s(1.7)
    t_plot = lps_qty.Time.s(0.25)
    t = np.arange(0, duration.get_s(), 1 / fs.get_hz())

    f0 = 1000.0           # Sine frequency [Hz]
    f_zoom = 20.0
    x = np.sin(2 * np.pi * f0 * t)

    # speeds_ms = [lps_qty.Speed.kt(s) for s in np.arange(-15, 20, 5)]
    speeds_ms = [lps_qty.Speed.m_s(s) for s in np.arange(-100, 100, 5)]
    sound_speed = lps_qty.Speed.m_s(1500.0)

    y1 = lps_propag_rsp.apply_doppler_by_block(x, speeds_ms, sound_speed)
    y2 = lps_propag_rsp.apply_doppler_block_crossfade(x, speeds_ms, sound_speed, fs)
    y3 = lps_propag_rsp.apply_doppler_by_sample(x, speeds_ms, sound_speed, fs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, sig, title in zip(
        axes,
        [x, y1],
        ["Original signal", "After apply_doppler"]
    ):
        f, tt, Sxx = scipy.signal.spectrogram(
            sig,
            fs=fs.get_hz(),
            nperseg=4096,
            noverlap=2048,
            mode="magnitude"
        )

        ax.pcolormesh(tt, f, Sxx, shading="gouraud")
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        # ax.set_ylim(0, 1000)
        ax.set_ylim(f0 - f_zoom, f0 + f_zoom)

    axes[0].set_ylabel("Frequency [Hz]")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_doppler.png"))


    n_offset_plot = int(t_offset_plot * fs)
    n_plot = int((t_offset_plot + t_plot) * fs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharey=True)

    axes[0].plot(t[n_offset_plot:n_plot], x[n_offset_plot:n_plot])
    axes[0].set_title("Original signal (time domain)")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t[n_offset_plot:n_plot], y1[n_offset_plot:n_plot])
    axes[1].set_title("After apply_doppler (time domain)")
    axes[1].set_xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_doppler_time.png"))
    plt.close()

    lps_signal.save_normalized_wav(x, fs, os.path.join(output_dir, "x.wav"))
    lps_signal.save_normalized_wav(y1, fs, os.path.join(output_dir, "y1.wav"))
    lps_signal.save_normalized_wav(y2, fs, os.path.join(output_dir, "y2.wav"))
    lps_signal.save_normalized_wav(y3, fs, os.path.join(output_dir, "y3.wav"))

if __name__ == "__main__":
    main()
