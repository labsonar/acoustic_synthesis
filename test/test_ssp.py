"""Test for get ssp based on WOA18 Data Access (WOA18)

Data available at:
    https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/
    https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA/temperature/csv/decav/0.25/woa18_t_decav_0.25_csv.tar.gz
    https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA/salinity/csv/decav/0.25/woa18_s_decav_0.25_csv.tar.gz

    https://services.data.shom.fr/geonetwork/WS/api/records/HOM_GEOL_SEDIM_MONDIALE.xml
    https://services.data.shom.fr/INSPIRE/telechargement/prepackageGroup/SEDIM_MONDIALE_PACK_DIFF_DL/prepackage/SEDIM_MONDIALE/file/SEDIM_MONDIALE.7z
"""
import os
import tqdm
import pandas as pd
import numpy as np

import gsw
import geopandas as gpd
import shapely.geometry as sgeo

import lps_synthesis.database.scenario as syndb_scenario
import lps_synthesis.scenario.dynamic as lps_sce_dyn
import lps_synthesis.propagation.layers as syn_lay

sedim_file = "/data/ambiental/SEDIM_MONDIALE/GML/SR.SeaBedAreaWorld.gml"
data = gpd.read_file(sedim_file)

# Codes de l’attribut TYPE_VALEU -> table 4.2.2 Descriptif de contenu du produit externe Mai 2021
SHOM_NAME_TO_ENUM = {
    # ROCHE [NFRoche] -> Rocha
    "NFRoche": syn_lay.SeabedType.BASALT,

    # CAILLOUTIS [NFC; NFCG; NFCS; NFCV] -> Cascalho grosso
    "NFC": syn_lay.SeabedType.GRAVEL,
    "NFCG": syn_lay.SeabedType.GRAVEL,
    "NFCS": syn_lay.SeabedType.GRAVEL,
    "NFCV": syn_lay.SeabedType.GRAVEL,

    # Graviers [NFG; NFGC; NFGS; NFGV] -> Cascalho
    "NFG": syn_lay.SeabedType.GRAVEL,
    "NFGC": syn_lay.SeabedType.GRAVEL,
    "NFGS": syn_lay.SeabedType.GRAVEL,
    "NFGV": syn_lay.SeabedType.GRAVEL,

    # Sables [NFS; NFSC; NFSG; NFSGV; NFSV; NFSSi] -> Areia
    "NFS": syn_lay.SeabedType.SAND,
    "NFSC": syn_lay.SeabedType.SAND,
    "NFSG": syn_lay.SeabedType.SAND,
    "NFSGV": syn_lay.SeabedType.SAND,
    "NFSV": syn_lay.SeabedType.SAND,
    "NFSSi": syn_lay.SeabedType.SAND,

    # Sables fins [NFSF; NFSFC; NFSFSi; NFSFV] -> Areia fina
    "NFSF": syn_lay.SeabedType.SAND,
    "NFSFC": syn_lay.SeabedType.SAND,
    "NFSFSi": syn_lay.SeabedType.SAND,
    "NFSFV": syn_lay.SeabedType.SAND,

    # Silts [NFSi ; NFSiA] -> Siltes
    "NFSi": syn_lay.SeabedType.SILT,
    "NFSiA": syn_lay.SeabedType.SILT,

    # Argiles [NFASi ; NFA] -> Argila
    "NFASi": syn_lay.SeabedType.CLAY,
    "NFA": syn_lay.SeabedType.CLAY,

    # Vases [NFV ; NFVC ; NFVG ; NFVS ; NFVSF] -> Lama (argila)
    "NFV": syn_lay.SeabedType.CLAY,
    "NFVC": syn_lay.SeabedType.CLAY,
    "NFVG": syn_lay.SeabedType.CLAY,
    "NFVS": syn_lay.SeabedType.CLAY,
    "NFVSF": syn_lay.SeabedType.CLAY,
}

def get_seabed(lat: float, lon: float):
    """
    Retorna o tipo de fundo marinho baseado no GML do SHOM.
    """
    point = sgeo.Point(lon, lat)

    match = data[data.contains(point)]

    if match.empty:

        # return None, None, True
        gdf = data.to_crs(3857)
        point_3857 = gpd.GeoSeries([point], crs=4326).to_crs(3857).iloc[0]

        gdf["dist"] = gdf.geometry.distance(point_3857)

        nearest = gdf.sort_values("dist").iloc[0]

        nearest_point = nearest.geometry.boundary.interpolate(
            nearest.geometry.boundary.project(point_3857)
        )
        # nearest_lonlat = gpd.GeoSeries([nearest_point], crs=3857).to_crs(4326).iloc[0]

        code = nearest["name"].strip()
        aprox = True

    else:
        row = match.iloc[0]
        code = row["name"].strip()
        aprox = False

    return code, SHOM_NAME_TO_ENUM[code], aprox

def get_ssp(lat, lon, month):
    # salinity_dir = "/data/ambiental/woa18_s_decav_0.25_csv"
    # temperature_dir = "/data/ambiental/woa18_t_decav_0.25_csv"
    base_dir = "/data/ambiental/woa18"

    salinity_file = os.path.join(base_dir, f"s{month:02.0f}.csv")
    temperature_file = os.path.join(base_dir, f"t{month:02.0f}.csv")

    df_s = pd.read_csv(salinity_file)
    df_t = pd.read_csv(temperature_file)

    df_s["dist"] = np.sqrt((df_s.iloc[:, 0] - lat)**2 +
                           (df_s.iloc[:, 1] - lon)**2)

    idx_s = df_s["dist"].idxmin()

    df_t["dist"] = np.sqrt((df_t.iloc[:, 0] - lat)**2 +
                           (df_t.iloc[:, 1] - lon)**2)

    idx_t = df_t["dist"].idxmin()

    row_s = df_s.loc[idx_s]
    row_t = df_t.loc[idx_t]

    sal_values = row_s[2:-1].values.astype(float)
    temp_values = row_t[2:-1].values.astype(float)

    N = min(len(sal_values), len(temp_values))
    sal_values = sal_values[:N]
    temp_values = temp_values[:N]

    valid_either = ~(np.isnan(sal_values) & np.isnan(temp_values))

    max_valid_idx = np.where(valid_either)[0].max()

    sal_values = sal_values[:max_valid_idx + 1]
    temp_values = temp_values[:max_valid_idx + 1]

    valid_both = ~np.isnan(sal_values) & ~np.isnan(temp_values)

    salt_profile = sal_values[valid_both]
    temp_profile = temp_values[valid_both]

    # idx = min(np.where(np.isnan(sal_values))[0][0], np.where(np.isnan(temp_values))[0][0])
    # salt_profile = sal_values[:idx]
    # temp_profile = temp_values[:idx]

    depth = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500]

    depth = depth[:len(sal_values)]
    depth = np.array(depth)
    depth = depth[valid_both]

    # depth = depth[:len(salt_profile)]
    # print("salt_profile: ", salt_profile)
    # print("temp_profile: ", temp_profile)

    pres = gsw.p_from_z(depth, lat)

    svp = gsw.sound_speed(salt_profile, temp_profile, pres)

    # print("pres: ", pres)
    # print("svp: ", svp)

    # print("t diff: ", df_t.iloc[idx_t, 0] - lat, df_t.iloc[idx_t, 1] - lon)
    # print("s diff: ", df_s.iloc[idx_s, 0] - lat, df_s.iloc[idx_s, 1] - lon)

    return (
        max(abs(df_t.iloc[idx_t, 0] - lat), abs(df_s.iloc[idx_s, 0] - lat)),
        max(abs(df_t.iloc[idx_t, 1] - lon), abs(df_s.iloc[idx_s, 1] - lon)),
        depth[-1],
        row_s.iloc[0],
        row_s.iloc[1]
    )


def main():

    month = 14

    for local in syndb_scenario.Local:
        p = local.get_point()

        code, seabed, aprox = get_seabed(p.latitude.get_deg(), p.longitude.get_deg())

        lat_diff, lon_diff, n_depths, lat_cor, lon_cor = get_ssp(p.latitude.get_deg(), p.longitude.get_deg(), month)


        print(local.name, ", ", p.latitude.get_deg(), ", ", p.longitude.get_deg(),", ", code, ", ", seabed, ", ", aprox, ", ", lat_diff, ", ", lon_diff, ", ", n_depths, ", ", lat_cor, ", ", lon_cor)


    # probe = {
    #     # "Moraine": {
    #     #     "Andfjorden": [69.214223, 16.482152],
    #     #     "Glaciar de Pine Island": [-70.625, -95.875],
    #     # },
    #     # "Chalk": {
    #     #     "Ekofisk": [56.802237, 2.708739],
    #     #     "Ontong Java Plateau": [-7.875, 159.625]
    #     # },
    #     # "Limestone": {
    #     #     "Mar adriatico": [43.648165, 14.391513],
    #     #     "Tanpa Bay": [27.585375, -82.883546]
    #     # }
    #     # "Silt": {
    #     #     "Baía de Bengala": [16.303433, 84.368654],
    #     #     "São Tomé e Príncipe": [0.219582, 3.826086],
    #     # },
    #     # "Deep": {
    #     #     "Chilean coast": [-43.625, -89.875],
    #     #     "Japan": [33.625, 150.125],
    #     #     "North Pacific": [14.375, -30.125],
    #     # }
    #     "test": {
    #         # "ANTIGUA_BARBUDA": [17.369162, -61.814704],
    #         # "ISLA_LAVA": [-14.520952, 47.558479]
    #         "CHILE": [-47.256955, -75.003445]
    #     }

    # }

    # for seabed_type, places in probe.items():
    #     for location, (lat, lon) in places.items():

    #         code, seabed, aprox = get_seabed(lat, lon)

    #         lat_diff, lon_diff, depth, lat_cor, lon_cor = get_ssp(lat, lon, month)

    #         print(f"{seabed_type}, {location}, {code}, {seabed}, {aprox}, {code}, {lat_diff}, {lon_diff}, {depth}, {lat_cor}, {lon_cor}")


if __name__ == "__main__":
    main()




    # latitudes = np.arange(-72, 90, 2)
    # longitudes = np.arange(-175, 170, 2)

    # for lat in tqdm.tqdm(latitudes):
    #     for lon in tqdm.tqdm(longitudes):
    # for lat in latitudes:
    #     for lon in longitudes:
    #         print(lat, ",", lon)

    #         code, seabed, aprox = get_seabed(lat, lon)

    #         if aprox:
    #             print("\t", aprox)
    #             continue

    #         lat_diff, lon_diff, n_depths, lat_cor, lon_cor = get_ssp(lat, lon, month)

    #         if abs(lat_diff) > 1 or abs(lon_diff) > 1:
    #             print("\t diff: ", lat_diff, ", ", lon_diff, ", ", n_depths, ", ", lat_cor, ", ", lon_cor)
    #             continue

    #         print(lat, ", ", lon, ", ", seabed, ", ", n_depths)

    # MORAINE
    # Local.BARENTS_SEA - https://www.lyellcollection.org/doi/10.1144/SP505-2019-82
    # Andfjorden - https://www.sciencedirect.com/science/article/pii/S0025322715000304
    # Pine Island Trough
    #Drake passage? -60.125	-45.375 -> -48.625	-35.125
    # Ekofisk, Tor and Hod Formations in the Danish sector of the North Sea https://www.earthdoc.org/content/journals/10.1111/j.1365-2478.2007.00622.x


    # Chalk
    # Ekofisk - https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JB008564
    # Ontong Java Plateau - https://agupubs.onlinelibrary.wiley.com/doi/10.1029/JB083iB01p00283


    # LIMESTONE
    # Mar adriatico 43º35'39.5200''N, 14º20'44.34''E https://www.sciencedirect.com/science/article/pii/S0264817215001099
    # The Bahama Banks - https://www.researchgate.net/publication/339373858_The_abyssal_giant_pockmarks_of_the_Black_Bahama_Escarpment_Relations_between_structures_fluids_and_carbonate_physiography

