#!/usr/bin/env python3
import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.channel_description as lps_desc
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.models as lps_models
import lps_synthesis.propagation.channel_response as lps_channel_rsp

def _run_comparison(
    sensor_depth: lps_qty.Distance,
    source_depth: lps_qty.Distance,
    local_depth: lps_qty.Distance,
    max_distance: lps_qty.Distance,
    output_dir: str,
    sample_frequency: lps_qty.Frequency,
    zoom_samples: int = 400,
):

    os.makedirs(output_dir, exist_ok=True)

    desc = lps_desc.Description()
    desc.add(lps_qty.Distance.m(0), lps_layer.Water(lps_qty.Speed.m_s(1500)))
    desc.add(local_depth - lps_qty.Distance.m(1), lps_layer.Water(lps_qty.Speed.m_s(1500)))
    desc.add(local_depth, lps_layer.BottomType.SAND)

    query = lps_models.QueryConfig(
        sample_frequency=sample_frequency,
        description=desc,
        source_depths=[source_depth],
        sensor_depth=sensor_depth,
        max_distance=max_distance,
        max_distance_points=500,
        frequency_range=None,
    )

    responses = {}
    timings = {}

    for model_type in lps_models.Type:
        print(f"Running model: {model_type.name}")

        model = model_type.build_model()

        t0 = time.perf_counter()
        rsp_f = model.compute_frequency_response(query)
        t1 = time.perf_counter()

        rsp_t = lps_channel_rsp.TemporalResponse.from_spectral(rsp_f)

        responses[model_type.name] = (rsp_f, rsp_t)
        timings[model_type.name] = t1 - t0

    ref_rsp_f, _ = next(iter(responses.values()))
    freqs = np.array([f.get_hz() for f in ref_rsp_f.frequencies])

    id_name = (
        f"source[{source_depth}]_"
        f"sensor[{sensor_depth}]_"
        f"depth[{local_depth}]_"
        f"distance[{max_distance}]"
    )


    fig, axs = plt.subplots(5, 1, figsize=(9, 9))
    fig.suptitle(id_name.replace("_", " | "))

    i0, i1 = None, None

    for model_name, (rsp_f, rsp_t) in responses.items():

        h_f = rsp_f.h_f_tau[-1, -1, :]
        h_t = rsp_t.h_t_tau[-1, -1, :]
        t = [t.get_s() for t in rsp_t.get_time_axis()]

        axs[0].plot(freqs, np.real(h_f), label=model_name)
        axs[1].plot(freqs, np.imag(h_f), label=model_name)
        axs[2].plot(freqs, -20 * np.log10(np.abs(h_f)), label=model_name)
        axs[3].plot(t, h_t, label=model_name)

        if i0 is None or i1 is None:
            peak = np.argmax(np.abs(h_t))
            start_offset = zoom_samples * 0.1
            end_offset = zoom_samples * 0.9
            i0 = int(max(0, peak - start_offset))
            i1 = int(min(len(h_t), peak + end_offset))

        axs[4].plot(t[i0:i1], h_t[i0:i1], label=model_name)

    # Labels
    axs[0].set_ylabel("Re{H}")
    axs[1].set_ylabel("Im{H}")
    axs[2].set_ylabel("|H| (dB)")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[3].set_ylabel("h(t)")
    axs[3].set_xlabel("Time (s)")
    axs[4].set_ylabel("h(t) zoom")
    axs[4].set_xlabel("Time (s)")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f"comparison_{id_name}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)

    print("\nExecution time (compute_frequency_response):")
    for name, t in sorted(timings.items(), key=lambda x: x[1]):
        print(f"{name:>12s}: {t:8.3f} s")


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Compare acoustic propagation models"
    )

    parser.add_argument("--sensor-depth",type=float,default=None,help="Sensor depth [m]")
    parser.add_argument("--source-depth", type=float, default=None, help="Source depth [m]")
    parser.add_argument("--local-depth", type=float, default=50000, help="Local depth [m]")
    parser.add_argument("--distance", type=float, default=100, help="Propagation distance [m]")

    parser.add_argument("--fs", type=float, default=16e3, help="Sample frequency [Hz]")
    parser.add_argument("--zoom-samples", type=int, default=400, help="Number of samples in zoom")
    parser.add_argument("--output-dir", default="result/model_comparison", help="Output directory")

    return parser.parse_args()

if __name__ == "__main__":

    args = _parse_args()

    _run_comparison(
        sensor_depth=lps_qty.Distance.m(args.sensor_depth or args.local_depth/2),
        source_depth=lps_qty.Distance.m(args.source_depth or args.local_depth/2),
        local_depth=lps_qty.Distance.m(args.local_depth),
        max_distance=lps_qty.Distance.m(args.distance),
        output_dir=args.output_dir,
        sample_frequency=lps_qty.Frequency.hz(args.fs),
        zoom_samples=args.zoom_samples
    )
