import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.channel_description as lps_desc
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.models as lps_models
import lps_synthesis.propagation.channel as lps_channel

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

    responses = {}
    timings = {}

    # for model_type in lps_models.Type:
    for model_type in [lps_models.Type.TRACEO]:
        print(f"Running model: {model_type.name}")

        model = model_type.build_model()

        t0 = time.perf_counter()
        channel = lps_channel.PredefinedChannel.SPHERICAL.get_channel(model=model)
        responses[model_type.name] = channel.get_ir()
        t1 = time.perf_counter()
        timings[model_type.name] = t1 - t0

    fig, axs = plt.subplots(2, 1, figsize=(9, 9))

    i0, i1 = None, None

    for model_name, rsp_t in responses.items():

        h_t = rsp_t.h_t_tau[-1, -1, :]
        t = [t.get_s() for t in rsp_t.get_time_axis()]

        axs[0].plot(t, h_t, label=model_name)

        if i0 is None or i1 is None:
            peak = np.argmax(np.abs(h_t))
            start_offset = zoom_samples * 0.1
            end_offset = zoom_samples * 0.9
            i0 = int(max(0, peak - start_offset))
            i1 = int(min(len(h_t), peak + end_offset))

        axs[1].plot(t[i0:i1], h_t[i0:i1], label=model_name)

    # Labels
    axs[0].set_ylabel("h(t)")
    axs[0].set_xlabel("Time (s)")
    axs[1].set_ylabel("h(t) zoom")
    axs[1].set_xlabel("Time (s)")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    fname = "comparison_spherical.png"
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
