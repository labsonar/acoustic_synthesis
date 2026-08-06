#!/usr/bin/env python3
"""
Generate a table with the environmental characteristics of every
(Location, Season) combination.
"""

import pandas as pd

import lps_synthesis.database.scenario as lps_sce
import lps_synthesis.environment.acoustic_site as lps_site

def main():

    prospector = lps_site.AcousticSiteProspector()

    rows = []

    for local in lps_sce.Location:

        point = local.get_point()

        # informações independentes da estação
        seabed, _ = prospector.seabed_prospector.get(point)
        depth = prospector.depth_prospector.get(point)

        for season in lps_site.Season:

            rain = prospector.environment_prospector.get_rain(
                point=point,
                season=season,
            )

            sea = prospector.environment_prospector.get_seastate(
                point=point,
                season=season,
            )

            rows.append({
                "LOCAL": local.name,
                "LOCAL_NAME": local.to_string(),
                "SEASON": season.name,
                "LATITUDE_DEG": point.latitude.get_deg(),
                "LONGITUDE_DEG": point.longitude.get_deg(),
                "SEABED": seabed.name,
                "DEPTH_M": depth.get_m(),
                "RAIN": rain.name,
                "SEA_STATE": sea,
            })

    df = pd.DataFrame(rows)

    df.to_csv("./result/acoustic_scenario_info.csv", index=False)

    print(df)
    print(f"\nSaved {len(df)} rows to acoustic_scenario_info.csv")


if __name__ == "__main__":
    main()