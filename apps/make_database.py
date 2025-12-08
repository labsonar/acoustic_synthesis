"""Make a new database
"""
import os
import argparse

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.database as syndb

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
        "--sensitivity",
        type=float,
        default=-165.0,
        help="Hydrophone sensitivity in dB re V/μPa (default: -165)",
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
        default=1.0,
        help="Step interval in seconds (default: 1)",
    )

    parser.add_argument(
        "--simulation-steps",
        type=int,
        default=10,
        help="Number of simulation steps (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        default="./result",
        help="Directory to save results (default: ./result)",
    )

    args = parser.parse_args()

    sonar = lps_sonar.Sonar.hydrophone(sensitivity=lps_qty.Sensitivity.db_v_p_upa(args.sensitivity))
    sample_frequency = lps_qty.Frequency.khz(args.sample_frequency)
    step_interval = lps_qty.Time.s(args.step_interval)
    simulation_steps = args.simulation_steps

    if args.dataset == "toy":
        dataset = syndb.ToyDatabase(n_samples=1)
    else:
        dataset = syndb.OlocumDatabase(n_ships=1, n_scenarios=1, n_samples=1)

    output_dir = os.path.join("./result", args.dataset)

    wav_dir = os.path.join(output_dir, "data")

    dataset.export(output_dir=output_dir)
    dataset.synthesize(output_dir=wav_dir,
                       sonar=sonar,
                       sample_frequency=sample_frequency,
                       step_interval=step_interval,
                       simulation_steps=simulation_steps)

if __name__ == "__main__":
    _main()
