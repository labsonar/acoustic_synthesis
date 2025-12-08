"""
Simple app to run syn_sites.prospect_local function, print and export the result
"""
import os
import argparse

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dyn
import lps_synthesis.propagation.layers as syn_lay
import lps_synthesis.environment.acoustic_site as syn_sites
import lps_synthesis.database.scenario as syn_scenario


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Prospector de áreas oceânicas para simulação acústica."
    )

    # --- Localização ---
    parser.add_argument(
        "--location",
        type=str,
        help=f"Nome do enum syn_scenario.Location. Opções: "
             f"{', '.join([loc.name for loc in syn_scenario.Location])}"
    )

    parser.add_argument("--lat", type=float, help="Latitude em graus.")
    parser.add_argument("--lon", type=float, help="Longitude em graus.")

    # --- Seabed opcional ---
    parser.add_argument(
        "--desired_seabed",
        type=str,
        default=None,
        help="Tipo de fundo desejado. Opções: "
             f"{', '.join([s.name for s in syn_lay.SeabedType])} (default: None)",
    )

    # --- Tolerâncias ---
    parser.add_argument(
        "--dist",
        type=float,
        help="Distância única (km) aplicada para lat e lon (default: 100 km)",
    )

    parser.add_argument("--dist_lat", type=float, help="Distância latitudinal em km.")
    parser.add_argument("--dist_lon", type=float, help="Distância longitudinal em km.")

    # --- Profundidade ---
    parser.add_argument(
        "--max_depth_dist",
        type=float,
        default=200,
        help="Distância máxima de profundidade (m). Default = 200 m.",
    )

    return parser.parse_args()


def _main():
    args = _parse_args()

    if args.location is not None:
        try:
            loc_enum = syn_scenario.Location[args.location]
        except KeyError as e:
            raise ValueError(f"Invalid location: {args.location}") from e

        center = loc_enum.get_point()

    else:
        if args.lat is None or args.lon is None:
            raise ValueError("If you don't use --location, you must provide --lat and --lon.")

        center = lps_dyn.Point.deg(args.lat, args.lon)

    if args.dist is not None:
        dist_lat = lps_qty.Distance.km(args.dist)
        dist_lon = lps_qty.Distance.km(args.dist)
    else:
        dist_lat = lps_qty.Distance.km(args.dist_lat) if args.dist_lat else lps_qty.Distance.km(100)
        dist_lon = lps_qty.Distance.km(args.dist_lon) if args.dist_lon else lps_qty.Distance.km(100)

    if args.desired_seabed is None:
        desired_seabed = None
    else:
        try:
            desired_seabed = syn_lay.SeabedType[args.desired_seabed]
        except KeyError as e:
            raise ValueError(f"Invalid seabed type: {args.desired_seabed}") from e

    max_depth_dist = lps_qty.Distance.m(args.max_depth_dist)

    df = syn_sites.prospect_local(
        center_point=center,
        dist_lat=dist_lat,
        dist_lon=dist_lon,
        desired_seabed=desired_seabed,
        max_depth_dist=max_depth_dist,
    )

    print("\n=== RESULTADO DO PROSPECTOR ===\n")
    print(df)

    os.makedirs("./result", exist_ok=True)
    output_path = "./result/prospector_site.csv"
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    _main()
