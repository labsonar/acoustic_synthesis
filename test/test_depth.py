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

import xarray as xr

ds = xr.open_dataset("/data/ambiental/etopo/etopo_2022_30.nc")
# ds = xr.open_dataset("/data/ambiental/etopo/etopo_2022.nc")
elev = ds["z"]  # nome da variável no ETOPO

def get_depth(lat, lon):
    #https://data.noaa.gov/metaview/page?xml=NOAA/NESDIS/NGDC/MGG/DEM//iso/xml/etopo_2022.xml&view=getDataView&header=none
    #60 arc-second bedrock elevation netCDF
    depth = elev.interp(lat=lat, lon=lon, method="linear").item()
    return depth  # negativo = mar


def main():

    data = []

    for local in syndb_scenario.Location:
        p = local.get_point()

        depth = get_depth(p.latitude.get_deg(), p.longitude.get_deg())

        data.append({
            "local": str(local),
            "depth": depth,
        })

    df = pd.DataFrame(data)
    print(df)

    # df.to_csv("./result/depths.csv")


if __name__ == "__main__":
    main()
