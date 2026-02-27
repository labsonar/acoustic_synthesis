"""Make a new database
"""
import os
import argparse
import tqdm
import numpy as np
import random

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig
import lps_utils.utils as lps_utils
import lps_ml.datasets as ml_db
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.database as syndb
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.database.dynamic as syndb_dyn
import lps_synthesis.scenario.noise_source as lps_ns

import memory_profiler


def _main():
    parser = argparse.ArgumentParser(
        description="Synthetic database generator for underwater acoustic test."
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Select the number of samples in dataset. (default: 250)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set seed. (default: 42)",
    )

    parser.add_argument(
        "--output-dir",
        default="/data/datatest",
        help="Directory to save results (default: /data/datatest)",
    )

    args = parser.parse_args()
    output_dir = args.output_dir

    cargo_dir = os.path.join(output_dir, "cargo")
    service_dir = os.path.join(output_dir, "service")
    env_dir = os.path.join(output_dir, "env")

    os.makedirs(cargo_dir, exist_ok=True)
    os.makedirs(service_dir, exist_ok=True)
    os.makedirs(env_dir, exist_ok=True)

    sample_frequency = lps_qty.Frequency.khz(16)
    simulation_time = lps_qty.Time.s(30)
    step_interval = lps_qty.Time.s(0.2)
    simulation_steps = int(simulation_time/step_interval)
    n_samples=int(sample_frequency * simulation_time)

    ### ENVIRONMENT

    for seed_offset in range(args.n_samples):

        filename = os.path.join(env_dir, f"{seed_offset}.wav")

        if os.path.exists(filename):
            continue

        environment = lps_env.Environment.random(seed=args.seed + seed_offset)

        noise = environment.generate_bg_noise(
            n_samples=n_samples,
            fs=int(sample_frequency.get_hz())
        )

        lps_sig.save_normalized_wav(
                noise,
                sample_frequency,
                filename
            )



    ### Cargo
    cargo_ships = [
        lps_ns.ShipType.BULKER,
        lps_ns.ShipType.CONTAINERSHIP,
        lps_ns.ShipType.VEHICLE_CARRIER,
        lps_ns.ShipType.TANKER
    ]
    service_ships = [
        lps_ns.ShipType.FISHING,
        lps_ns.ShipType.RECREATIONAL,
        lps_ns.ShipType.TUG,
        lps_ns.ShipType.DREDGER,
    ]
    rng = random
    for seed_offset in range(2 * args.n_samples):

        if seed_offset < args.n_samples:

            ship = lps_ns.Ship.by_type(
                ship_type=random.choice(cargo_ships),
                seed=args.seed + seed_offset
            )
            out_dir = cargo_dir

        else:

            ship = lps_ns.Ship.by_type(
                ship_type=random.choice(service_ships),
                seed=args.seed + seed_offset
            )
            out_dir = service_dir

        filename = os.path.join(out_dir, f"{seed_offset}.wav")

        if os.path.exists(filename):
            continue

        ship.move(step_interval=step_interval, n_steps=simulation_steps)

        compiler = lps_ns.NoiseCompiler(
            noise_containers=[ship],
            fs=sample_frequency,
            parallel=False
        )

        for signal, _, _ in compiler:

            lps_sig.save_normalized_wav(
                    signal,
                    sample_frequency,
                    filename
                )
            break




@memory_profiler.profile
def _run():
    _main()

if __name__ == "__main__":
    _run()
