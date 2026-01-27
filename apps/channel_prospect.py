import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")

import lps_synthesis.database as lps_db
import lps_synthesis.environment.acoustic_site as lps_as
import lps_synthesis.propagation.models as lps_models


def run_case(location: lps_db.Location, model_name: str, outdir: str):

    print(f"[RUN] Location={location.name}, Model={model_name}")

    model = lps_models.Type[model_name].build_model()

    acoustic_scenario = lps_db.AcousticScenario(
        location,
        lps_as.Season.SPRING,
    )

    channel = acoustic_scenario.get_channel(model=model)
    ir = channel.get_ir()

    os.makedirs(outdir, exist_ok=True)

    filename = os.path.join(
        outdir,
        f"ir_{location.name.lower()}_{model_name}.png"
    )

    # ir.print_as_image(filename=filename)
    print(f"[OK] Saved {filename}")


def _main():
    parser = argparse.ArgumentParser(
        description="Test channel impulse responses"
    )

    parser.add_argument(
        "--location",
        type=str,
        choices=[loc.name for loc in lps_db.Location],
        help="Location enum name",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=[loc.name for loc in lps_models.Type],
        help="Propagation model",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all Location x Model combinations",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./result/channel_ir",
    )

    parser.add_argument(
        "--nproc",
        type=int,
        default=max(1, mp.cpu_count() - 2),
        help="Maximum number of parallel threads",
    )

    args = parser.parse_args()

    if args.all:

        _ = lps_as.AcousticSiteProspector()

        tasks = []
        with ThreadPoolExecutor(max_workers=args.nproc) as executor:

            for loc in lps_db.Location:
                for model_type in lps_models.Type:
                    future = executor.submit(
                        run_case,
                        loc,
                        model_type.name,
                        args.outdir
                    )
                    tasks.append(future)

            for f in as_completed(tasks):
                f.result()

    else:
        if args.location is None or args.model is None:
            parser.error("Either use --all or specify --location and --model")

        location = lps_db.Location[args.location]
        run_case(location, args.model, args.outdir)


if __name__ == "__main__":
    _main()
