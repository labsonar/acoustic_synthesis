import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")

import lps_synthesis.database as lps_db
import lps_synthesis.environment.acoustic_site as lps_as
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.models as lps_models


def run_case(location: lps_db.Location,
             model_name: str,
             season: lps_as.Season,
             outdir: str,
             plot_ir: bool = True):

    print(f"[RUN] Location={location.name}, Model={model_name}")

    model = lps_models.Type[model_name].build_model()

    acoustic_scenario = lps_db.AcousticScenario(
        location,
        season,
    )

    channel = acoustic_scenario.get_channel(model=model)

    if not plot_ir:
        return

    ir = channel.get_ir()

    os.makedirs(outdir, exist_ok=True)

    filename = os.path.join(
        outdir,
        f"ir_{location.name.lower()}_{model_name}_{season.name.lower()}.png"
    )

    ir.print_as_image(filename=filename)
    print(f"[OK] Saved {filename}")


def build_as_row(location, season, model_type, output_dir):

    acoustic_scenario = lps_db.AcousticScenario(
        location,
        season,
    )

    model = model_type.build_model()

    channel = acoustic_scenario.get_channel(model=model)
    env = acoustic_scenario.get_env()

    channel_filename = os.path.basename(channel._filename())

    desc = channel.query.description
    desc_filename = f"{location.name.lower()}_{season.name.lower()}.pkl"
    desc_path = os.path.join(output_dir, desc_filename)
    desc.save(desc_path)

    return {
        "LOCAL": location.name,
        "SEASON": season.name,
        "MODEL": model_type.name.lower(),
        "CHANNEL_FILENAME": channel_filename,
        "SENSOR_DEPTH": channel.query.sensor_depth.get_m(),
        "RAIN_VALUE": env.rain_value.value if isinstance(env.rain_value, lps_env.Rain) \
                                           else env.rain_value,
        "SEA_VALUE": env.sea_value.value if isinstance(env.sea_value, lps_env.Sea) \
                                         else env.sea_value,
        "SHIPPING_VALUE": env.shipping_value.value \
                                        if isinstance(env.shipping_value, lps_env.Shipping) \
                                        else env.shipping_value,
        "SEED": int(env.seed),
    }


def _main():
    parser = argparse.ArgumentParser(
        description="Test channel impulse responses"
    )

    parser.add_argument(
        "--season",
        type=str,
        choices=[season.name for season in lps_as.Season],
        help="Season enum name",
    )

    loc_names = ", ".join(loc.name for loc in lps_db.Location)

    parser.add_argument(
        "--location",
        type=str,
        metavar="{" + loc_names + "}[,{...}]",
        help=(
            "Location enum name or comma-separated list. "
            f"Valid values: {loc_names}"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=[loc.name for loc in lps_models.Type],
        help="Propagation model",
    )

    parser.add_argument(
        "--export-df",
        action="store_true",
        help="Export CSV with channel and environment metadata",
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

    if args.location is None:
        locations = list(lps_db.Location)
    else:
        try:
            locations = [
                lps_db.Location[name.strip()]
                for name in args.location.split(",")
            ]
        except KeyError as e:
            valid = ", ".join(loc.name for loc in lps_db.Location)
            raise ValueError(
                f"Invalid location '{e.args[0]}'. "
                f"Valid options are: {valid}"
            ) from e


    seasons = (
        [lps_as.Season[args.season]]
        if args.season is not None
        else list(lps_as.Season)
    )

    models = (
        [lps_models.Type[args.model]]
        if args.model is not None
        else list(lps_models.Type)
    )

    if args.export_df:
        rows = []

        output_dir = os.path.join(args.outdir, "acoustic_scenario")
        os.makedirs(output_dir, exist_ok=True)

        for season in seasons:
            for loc in locations:
                for model_type in models:
                    rows.append(build_as_row(loc, season, model_type, output_dir))

        df = pd.DataFrame(rows)

        csv_path = os.path.join(output_dir, "acoustic_scenario_info.csv")
        df.to_csv(csv_path, index=False)

        return

    # _ = lps_as.AcousticSiteProspector()
    # tasks = []
    # with ThreadPoolExecutor(max_workers=args.nproc) as executor:

    #     for season in seasons:
    #         for loc in locations:
    #             for model_type in models:
    #                 future = executor.submit(
    #                     run_case,
    #                     loc,
    #                     model_type.name,
    #                     season,
    #                     args.outdir,
    #                     False
    #                 )
    #                 tasks.append(future)

    #     for f in as_completed(tasks):
    #         f.result()


    for loc in locations:
        for model_type in models:
            for season in seasons:
                run_case(
                    loc,
                    model_type.name,
                    season,
                    args.outdir,
                    True
                )



if __name__ == "__main__":
    _main()
