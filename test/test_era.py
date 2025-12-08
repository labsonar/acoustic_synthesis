"""Test for get ssp based on WOA18 Data Access (WOA18)

Data available at:
    https://cds.climate.copernicus.eu/datasets

    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview
    Monthly averaged reanalysis
    Significant height of combined wind waves and swell
    Total precipitation
    2024
    all months
    NetCDF4
"""
import os
import tqdm
import pandas as pd
import numpy as np

import gsw
import geopandas as gpd
import shapely.geometry as sgeo
import matplotlib.pyplot as plt

import lps_synthesis.database.scenario as syndb_scenario
import lps_synthesis.scenario.dynamic as lps_sce_dyn
import lps_synthesis.propagation.layers as syn_lay
import lps_synthesis.environment.acoustic_site as syn_site
import lps_synthesis.environment.environment as syn_env

import xarray as xr


SEASON_MONTHS = {
    syn_site.Season.SUMMER: [12, 1, 2],
    syn_site.Season.AUTUMN: [3, 4, 5],
    syn_site.Season.WINTER: [6, 7, 8],
    syn_site.Season.SPRING: [9, 10, 11],
}


# ============================
# Douglas scale conversion
# ============================
import random
def hs_to_douglas(hs: float) -> int:
    """Convert significant wave height (m) to Douglas scale."""
    boundaries = [
        (0, 0.1, 0),
        (0.1, 0.5, 1),
        (0.5, 1.25, 2),
        (1.25, 2.5, 3),
        (2.5, 4.0, 4),
        (4.0, 6.0, 5),
        (6.0, 9.0, 6),
        (9.0, 14.0, 7),
        (14.0, 20.0, 8),
    ]

    for low, high, douglas in boundaries:
        if low <= hs < high:
            return douglas
    return random.randint(0,7)


def tp_to_seastate(tp: float) -> syn_env.Rain:
    """Convert significant wave height (m) to Douglas scale."""
    boundaries = [
        (0, 0.004, syn_env.Rain.NONE),
        (0.004, 0.008, syn_env.Rain.LIGHT),
        (0.008, 0.016, syn_env.Rain.MODERATE),
        (0.016, 0.032, syn_env.Rain.HEAVY),
    ]

    for low, high, douglas in boundaries:
        if low <= tp < high:
            return douglas
    return syn_env.Rain.VERY_HEAVY

DS_TP = xr.open_dataset("/data/ambiental/era5/data_stream-moda_stepType-avgad.nc")
DS_SWH = xr.open_dataset("/data/ambiental/era5/data_stream-wamd_stepType-avgua.nc")

def seasonal_mean_point(ds, var_name: str, months: list, lat: float, lon: float, mean: bool):
    """
    Retorna a média sazonal da variável ERA-5 interpolada para uma latitude/longitude.
    """
    da = ds[var_name]

    # Filtrando apenas os meses da estação
    time_index = da["valid_time"].to_index()
    seasonal_mask = time_index.month.isin(months)
    da_filtered = da.sel(valid_time=seasonal_mask)

    # Interpolando para o ponto solicitado
    da_point = da_filtered.interp(latitude=lat, longitude=lon)

    # Média final
    if mean:
        return float(da_point.mean().values)
    return float(da_point.max().values)


def get_seastate(season: syn_site.Season, point) -> int:
    """
    Retorna o estado do mar (Douglas) para uma estação e ponto (lat/lon).
    """
    lat = point.latitude.get_deg()
    lon = (360 + point.longitude.get_deg()) % 360
    months = SEASON_MONTHS[season]

    # Altura significativa média da estação
    hs_mean = seasonal_mean_point(DS_SWH, "swh", months, lat, lon, True)

    # Converte para o mar de Douglas
    return hs_to_douglas(hs_mean)


def get_rain(season: syn_site.Season, point) -> float:
    """
    Retorna a precipitação média em mm/h para a estação e ponto.
    """
    lat = point.latitude.get_deg()
    lon = (360 + point.longitude.get_deg()) % 360
    months = SEASON_MONTHS[season]

    # tp do ERA-5 (m/day no produto mensal)
    tp_m = seasonal_mean_point(DS_TP, "tp", months, lat, lon, False)

    return tp_to_seastate(tp_m)

def main():

    data = []

    env_prospector = syn_site.EnvironmentProspector()

    for local in syndb_scenario.Location:
        p = local.get_point()

        for season in syn_site.Season:

            rain = env_prospector.get_rain(point = p, season = season)
            seastate = env_prospector.get_seastate(point = p, season = season)

            data.append({
                "local": str(local),
                "season": str(season),
                "rain": str(rain),
                "seastate": str(seastate),
            })

    df = pd.DataFrame(data)

    df.to_csv("./result/era.csv")

    print()
    print("############")
    print(df)

    count = df.groupby(['rain']).size().reset_index(name="Qty")
    print()
    print("############")
    print(count)

    count = df.groupby(['seastate']).size().reset_index(name="Qty")
    print()
    print("############")
    print(count)


    # data = []

    # for local in syndb_scenario.Location:
    #     p = local.get_point()

    #     for season in syn_site.Season:

    #         rain = get_rain(season, p)
    #         seastate = get_seastate(season, p)

    #         data.append({
    #             "local": str(local),
    #             "season": str(season),
    #             "rain": rain,
    #             "seastate": seastate,
    #         })

    # df = pd.DataFrame(data)
    # print(df)

    # df.to_csv("./result/era.csv")


    # tp = DS_TP["tp"]
    # tp_m = tp.values.reshape(-1)
    # plt.hist(tp_m, bins=30)
    # plt.yscale("log")
    # plt.xlabel("Rain (m)")
    # plt.ylabel("Count")
    # plt.title("Seasonal rain distribution")
    # plt.savefig("./result/era_tp.png")

    # thresholds = [0.005, 0.02, 0.05, 0.09]

    # count_lt_first = np.sum(tp_m < thresholds[0])
    # count_1_2 = np.sum((tp_m >= thresholds[0]) & (tp_m < thresholds[1]))
    # count_2_3 = np.sum((tp_m >= thresholds[1]) & (tp_m < thresholds[2]))
    # count_3_4 = np.sum((tp_m >= thresholds[2]) & (tp_m < thresholds[3]))
    # count_gt_last = np.sum(tp_m >= thresholds[3])

    # print(" < 0.005 :", count_lt_first)
    # print("0.005–0.02:", count_1_2)
    # print("0.02–0.05:", count_2_3)
    # print("0.05–0.09:", count_3_4)
    # print(" > 0.09 :", count_gt_last)



    # swh = DS_SWH["swh"]
    # swh_m = swh.values.reshape(-1)
    # plt.hist(swh_m, bins=30)
    # plt.yscale("log")
    # plt.savefig("./result/era_swh.png")

if __name__ == "__main__":
    main()
