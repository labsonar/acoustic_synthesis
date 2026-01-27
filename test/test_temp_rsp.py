#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig
import lps_synthesis.propagation.channel as lps_propag
import lps_synthesis.propagation.channel_response as lps_channel_rsp
import lps_synthesis.propagation.models as lps_models
import lps_synthesis.scenario.dynamic as lps_dyn
import lps_synthesis.scenario.noise_source as lps_ns


def main():

    output_dir = "./result/apply_test"
    os.makedirs(output_dir, exist_ok=True)

    fs = lps_qty.Frequency.khz(16)
    duration = lps_qty.Time.s(10)
    interval = lps_qty.Time.s(2)
    n_samples = int(fs * duration)
    r_start = 50.0
    r_end = 100.0

    channel = lps_propag.PredefinedChannel.SPHERICAL.get_channel(lps_models.Oases())

    # source_depth = channel.query.source_depths[-1]
    source_depth = lps_qty.Distance.m(20)

    # rng = np.random.default_rng(1234)
    # x = rng.standard_normal(n_samples) * 1800000/5

    ship = lps_ns.Ship(
            ship_id="Ship_1",
            propulsion=lps_ns.CavitationNoise(
                ship_type=lps_ns.ShipType.BULKER,
                cruise_rotacional_frequency = lps_qty.Frequency.rpm(80),
            )
        )

    ship.move(step_interval=interval, n_steps=int(duration/interval))

    x = ship.noise_sources[0].generate_noise(fs=fs)

    ref = lps_dyn.State()
    speeds = []

    for i in range(ship.get_n_steps()):
        speeds.append(ship[i].get_relative_speed(ref))

    x1 = lps_channel_rsp.apply_doppler_by_sample(
        input_data=x,
        speeds=speeds,
        sound_speed=lps_qty.Speed.m_s(1500),
        sample_frequency=fs
    )

    x2 = lps_channel_rsp.apply_doppler_by_block(
        input_data=x,
        speeds=speeds,
        sound_speed=lps_qty.Speed.m_s(1500),
    )

    x3 = lps_channel_rsp.apply_doppler_block_crossfade(
        input_data=x,
        speeds=speeds,
        sound_speed=lps_qty.Speed.m_s(1500),
        sample_frequency=fs
    )

    distances = np.linspace(r_start, r_end, 20)
    distances = [lps_qty.Distance.m(d) for d in distances]

    print("### channel.propagate")
    print("input_data: ", x.shape, " -> ", x.dtype)
    print("limits: ", x.min(), " -> ", x.max())
    print("source_depth: ", source_depth)
    print("distance[", len(distances), "]: ", distances)
    print("sample_frequency: ", fs)
    print("### channel.propagate")

    print("x1: ", x1.shape)
    print("x2: ", x2.shape)
    print("x3: ", x3.shape)

    y = channel.propagate(
        input_data=x,
        source_depth=source_depth,
        distance=distances,
        sample_frequency=fs
    )

    lps_sig.save_normalized_wav(
        x,
        fs,
        os.path.join(output_dir, "input.wav")
    )
    lps_sig.save_normalized_wav(
        x1,
        fs,
        os.path.join(output_dir, "doppler.wav")
    )
    lps_sig.save_normalized_wav(
        x2,
        fs,
        os.path.join(output_dir, "doppler_by_block.wav")
    )
    lps_sig.save_normalized_wav(
        x3,
        fs,
        os.path.join(output_dir, "doppler_block_crossfade.wav")
    )
    lps_sig.save_normalized_wav(
        y,
        fs,
        os.path.join(output_dir, "output_after_channel.wav")
    )

    spec_x = np.fft.rfft(x)
    spec_y = np.fft.rfft(y)
    freq = np.fft.rfftfreq(len(x), d=1/fs.get_hz())
    mag_db_x = 20 * np.log10(np.abs(spec_x) + 1e-12)
    mag_db_y = 20 * np.log10(np.abs(spec_y) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(freq, mag_db_x)
    plt.plot(freq, mag_db_y)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spectrum.png"), dpi=300)
    plt.close()

    fig, axs = plt.subplots(3, 1, figsize=(9, 9))
    t = np.arange(len(x)) / fs.get_hz()

    axs[0].plot(t, x)
    axs[1].plot(t, y)

    h_t = channel.get_ir().h_t_tau[-1, 0, :]
    t = np.arange(len(h_t)) / fs.get_hz()
    axs[2].plot(t, h_t)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    plt.savefig(os.path.join(output_dir, "time.png"), dpi=300)
    plt.close(fig)


    print("\nTest completed successfully!")
    print(f"Results saved in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
