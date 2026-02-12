import os

import pandas as pd

import lps_utils.quantities as lps_qty
import lps_synthesis.database.scenario as lps_sce
import lps_synthesis.environment.acoustic_site as lps_as

def _build_ssp_depth_summary() -> pd.DataFrame:

    ssp = lps_as.SSPProspector()

    rows = []

    for location in lps_sce.Location:
        point = location.get_point()

        for season in lps_as.Season:
            try:
                depths, _ = ssp.get(
                    point=point,
                    season=season,
                )

                min_depth = float(depths[0].get_m())
                max_depth = float(depths[-1].get_m())

                rows.append({
                    "LOCATION_ID": location.value,
                    "LOCATION_NAME": location.to_string("en_us"),
                    "SEASON": season.name,
                    "MIN_DEPTH_(m)": min_depth,
                    "MAX_DEPTH_(m)": max_depth,
                    "N_LEVELS": len(depths)
                })

            except Exception as e:
                rows.append({
                    "LOCATION_ID": location.value,
                    "LOCATION_NAME": location.to_string("en_us"),
                    "SEASON": season.name,
                    "MIN_DEPTH_(m)": None,
                    "MAX_DEPTH_(m)": None,
                    "N_LEVELS": None,
                    "ERROR": str(e)
                })

    df = pd.DataFrame(rows)

    return df


if __name__ == "__main__":

    output_dir = "./result/ssp_test"
    os.makedirs(output_dir, exist_ok=True)

    df = _build_ssp_depth_summary()

    # Mostrar completo no terminal
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\nSSP DEPTH SUMMARY\n")
    print(df.sort_values(["LOCATION_ID", "SEASON"]))

    df.to_csv(os.path.join(output_dir, "depths.csv"))
