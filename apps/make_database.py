"""Make a new database
"""
import os
import argparse
import tqdm

import lps_utils.quantities as lps_qty
import lps_utils.utils as lps_utils
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.database as syndb

import memory_profiler

def _main():
    parser = argparse.ArgumentParser(
        description="Synthetic database generator for underwater acoustic scenarios."
    )

    parser.add_argument(
        "--dataset",
        choices=["toy", "olocum"],
        default="toy",
        help="Select the dataset type: 'toy' or 'olocum'. (default: toy)",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Select the number of samples in dataset. (default: 1)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set seed. (default: 42)",
    )

    parser.add_argument(
        "--only-info",
        action="store_true",
        help="Only print database info",
    )

    parser.add_argument(
        "--only-plot",
        action="store_true",
        help="Only plot sample dynamics",
    )

    parser.add_argument(
        "--sensitivity",
        type=float,
        default=-200.0,
        help="Hydrophone sensitivity in dB re V/μPa (default: -180)",
    )

    parser.add_argument(
        "--sample-frequency",
        type=float,
        default=16.0,
        help="Sampling frequency in kHz (default: 16)",
    )

    parser.add_argument(
        "--step-interval",
        type=float,
        default=0.2,
        help="Step interval in seconds (default: 0.2)",
    )

    parser.add_argument(
        "--simulation-steps",
        type=int,
        default=150,
        help="Number of simulation steps (default: 150)",
    )

    parser.add_argument(
        "--env-att",
        type=float,
        default=0,
        help="Glocabal db attenuation of enviroment noise (default: 0)",
    )

    parser.add_argument(
        "--load",
        action="store_true",
        help="Load previolsy computed info",
    )

    parser.add_argument(
        "--force_override",
        action="store_true",
        help="Override wav files if they exists",
    )

    parser.add_argument(
        "--sample-index",
        type=str,
        default=None,
        help="Accepts single value, ranges and compositions (e.g. 3, 1-5, 1/3-6/10)",
    )

    parser.add_argument(
        "--limits",
        action="store_true",
        help="Compute global operational limits (distance and speed) without audio synthesis",
    )

    parser.add_argument(
        "--output-dir",
        default="./result",
        help="Directory to save results (default: ./result)",
    )

    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.dataset)

    if args.load:
        dataset = syndb.Database.load(output_dir)

    else:

        if args.dataset == "toy":
            dataset = syndb.ToyDatabase(n_samples=args.n_samples)
        else:
            dataset = syndb.OlocumDatabase(n_samples=args.n_samples, seed=args.seed)

        dataset.export(output_dir=output_dir)

    if args.only_info:
        print("############## Dataset ###############")
        print(dataset.to_df())
        print("############## Ship Catalog ###############")
        print(dataset.ship_catalog.to_df())
        print("############## Acoustic Scenario ###############")
        print(dataset.acoutic_scenario_catalog.to_df())

    elif args.limits:

        sonar = lps_sonar.Sonar.hydrophone(
            sensitivity=lps_qty.Sensitivity.db_v_p_upa(args.sensitivity)
        )

        sample_frequency = lps_qty.Frequency.khz(args.sample_frequency)
        step_interval = lps_qty.Time.s(args.step_interval)

        dataset.compute_limits(
            sonar=sonar,
            step_interval=step_interval,
            simulation_steps=args.simulation_steps,
            global_attenuation_db=args.env_att,
        )

    else:
        wav_dir = os.path.join(output_dir, "data")

        sonar = lps_sonar.Sonar.hydrophone(
            sensitivity=lps_qty.Sensitivity.db_v_p_upa(args.sensitivity)
        )
        sample_frequency = lps_qty.Frequency.khz(args.sample_frequency)
        step_interval = lps_qty.Time.s(args.step_interval)
        simulation_steps = args.simulation_steps

        if args.sample_index is not None:

            indices = lps_utils.parse_indices(args.sample_index)

            if not indices:
                raise ValueError("Invalid --sample-index format")

            for idx in tqdm.tqdm(indices, desc="Making samples", leave=False, ncols=120):
                dataset.synthesize_sample(
                    sample_index=idx,
                    output_dir=wav_dir,
                    sonar=sonar,
                    sample_frequency=sample_frequency,
                    step_interval=step_interval,
                    simulation_steps=simulation_steps,
                    global_attenuation_db=args.env_att,
                    only_plot=args.only_plot,
                    force_override=args.force_override
                )
        else:
            dataset.synthesize(
                output_dir=wav_dir,
                sonar=sonar,
                sample_frequency=sample_frequency,
                step_interval=step_interval,
                simulation_steps=simulation_steps,
                global_attenuation_db=args.env_att,
                only_plot=args.only_plot,
                force_override=args.force_override
            )



@memory_profiler.profile
def _run():
    _main()

if __name__ == "__main__":
    _run()
