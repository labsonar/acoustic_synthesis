import os
import enum
import math
import typing
import random
import tqdm

import numpy as np
import pandas as pd

import gsw
import geopandas as gpd
import shapely.geometry as sgeo
import xarray as xr

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as syn_lay
import lps_synthesis.propagation.channel_description as lps_desc
import lps_synthesis.propagation.channel as lps_channel
import lps_synthesis.scenario.dynamic as lps_dyn
import lps_synthesis.environment.environment as lps_env

class Season(enum.IntEnum):
    """Enum representing seasons of the year."""
    SUMMER = 1
    AUTUMN = 2
    WINTER = 3
    SPRING = 4

    def get_months(self, lat: lps_qty.Latitude) -> list[int]:
        """Return list of months for this season based on latitude sign."""
        _north_map = {
            Season.SUMMER: [6, 7, 8],
            Season.AUTUMN: [9, 10, 11],
            Season.WINTER: [12, 1, 2],
            Season.SPRING: [3, 4, 5],
        }

        _south_map = {
            Season.SUMMER: [12, 1, 2],
            Season.AUTUMN: [3, 4, 5],
            Season.WINTER: [6, 7, 8],
            Season.SPRING: [9, 10, 11],
        }
        if lat < 0:
            return _south_map[self]
        else:
            return _north_map[self]

    @staticmethod
    def rand() -> "Season":
        """Return a randomly selected season."""
        return random.choice(list(Season))

class SeabedProspector:
    """
    Query seabed types from the SHOM global sediment database.

    This class maintains a static cache of the SHOM dataset in two coordinate
    systems: EPSG:4326 (latitude/longitude) and EPSG:3857 (Web Mercator).
    The cache ensures that the dataset is loaded and reprojected only once,
    even if multiple SeabedProspector instances are created.

    The SHOM sediment data is publicly available from:

        Metadata:
            https://services.data.shom.fr/geonetwork/WS/api/records/HOM_GEOL_SEDIM_MONDIALE.xml

        Direct download:
            https://services.data.shom.fr/INSPIRE/telechargement/prepackageGroup/SEDIM_MONDIALE_PACK_DIFF_DL/prepackage/SEDIM_MONDIALE/file/SEDIM_MONDIALE.7z

    The required file is located inside the package:

        SEDIM_MONDIALE/GML/SR.SeaBedAreaWorld.gml
    """

    _cache_4326: typing.Optional[gpd.GeoDataFrame] = None
    _cache_3857: typing.Optional[gpd.GeoDataFrame] = None

    # Convertion based on Codes de l’attribut TYPE_VALEU
    # table 4.2.2 of Descriptif de contenu du produit externe Mai 2021
    _convert_map = {
        # ROCHE
        "NFRoche": syn_lay.SeabedType.BASALT,

        # CAILLOUTIS → GRAVEL
        "NFC": syn_lay.SeabedType.GRAVEL,
        "NFCG": syn_lay.SeabedType.GRAVEL,
        "NFCS": syn_lay.SeabedType.GRAVEL,
        "NFCV": syn_lay.SeabedType.GRAVEL,

        # GRAVIERS → GRAVEL
        "NFG": syn_lay.SeabedType.GRAVEL,
        "NFGC": syn_lay.SeabedType.GRAVEL,
        "NFGS": syn_lay.SeabedType.GRAVEL,
        "NFGV": syn_lay.SeabedType.GRAVEL,

        # SABLES → SAND
        "NFS": syn_lay.SeabedType.SAND,
        "NFSC": syn_lay.SeabedType.SAND,
        "NFSG": syn_lay.SeabedType.SAND,
        "NFSGV": syn_lay.SeabedType.SAND,
        "NFSV": syn_lay.SeabedType.SAND,
        "NFSSi": syn_lay.SeabedType.SAND,

        # FINE SANDS → SAND
        "NFSF": syn_lay.SeabedType.SAND,
        "NFSFC": syn_lay.SeabedType.SAND,
        "NFSFSi": syn_lay.SeabedType.SAND,
        "NFSFV": syn_lay.SeabedType.SAND,

        # SILTS → SILT
        "NFSi": syn_lay.SeabedType.SILT,
        "NFSiA": syn_lay.SeabedType.SILT,

        # CLAYS → CLAY
        "NFASi": syn_lay.SeabedType.CLAY,
        "NFA": syn_lay.SeabedType.CLAY,

        # MUD (argila) → CLAY
        "NFV": syn_lay.SeabedType.CLAY,
        "NFVC": syn_lay.SeabedType.CLAY,
        "NFVG": syn_lay.SeabedType.CLAY,
        "NFVS": syn_lay.SeabedType.CLAY,
        "NFVSF": syn_lay.SeabedType.CLAY,
    }

    def __init__(self, shom_file: str = "/data/ambiental/shom/SR.SeaBedAreaWorld.gml"):
        """
        Load SHOM sediment dataset and prepare cached GeoDataFrames.

        Parameters
        ----------
        shom_file : str
            Path to the SHOM sediment shapefile.
        """

        if SeabedProspector._cache_4326 is None:
            shom_data = gpd.read_file(shom_file)
            SeabedProspector._cache_4326 = shom_data.to_crs(4326)

        if SeabedProspector._cache_3857 is None:
            SeabedProspector._cache_3857 = SeabedProspector._cache_4326.to_crs(3857)

        # Bind instance views
        self.data_polar : gpd.GeoDataFrame = SeabedProspector._cache_4326
        self.data_cartesian : gpd.GeoDataFrame = SeabedProspector._cache_3857

    def get(
        self,
        point: lps_dyn.Point,
        tol: lps_qty.Distance = lps_qty.Distance.km(30)
    ) -> typing.Tuple[syn_lay.SeabedType, lps_qty.Distance]:
        """
        Return the seabed type at a given geographic point.

        The method first checks whether the point is *covered* by a polygon.
        If not, it computes the nearest polygon boundary in Web Mercator,
        verifies the real geographic distance, and returns its seabed type.

        Parameters
        ----------
        point : lps_dyn.Point
            Point with latitude and longitude.
        tol : lps_qty.Distance
            Maximum allowed distance to the nearest polygon boundary.

        Returns
        -------
        syn_lay.SeabedType
            The seabed type.
        lps_qty.Distance
            nearest polygon boundary distance.

        Raises
        ------
        ValueError
            If the nearest polygon boundary exceeds the tolerance.
        """

        # Create shapely point (lon, lat)
        sgeo_point = sgeo.Point(point.longitude.get_deg(), point.latitude.get_deg())

        # Check the polygon that contains the test point.
        mask = self.data_polar.covers(sgeo_point)
        match = self.data_polar.loc[mask]

        if not match.empty:
            code = match.iloc[0]["name"].strip()
            return SeabedProspector._convert_map[code], lps_qty.Distance.m(0)

        # Convert point to Web Mercator
        sgeo_point_cartesian = (
            gpd.GeoSeries([sgeo_point], crs=4326).to_crs(3857).iloc[0]
        )

        self.data_cartesian.loc[:, "dist"] = (
            self.data_cartesian.geometry.distance(sgeo_point_cartesian)
        )

        nearest = self.data_cartesian.sort_values("dist").iloc[0]
        boundary = nearest.geometry.boundary
        projected = boundary.project(sgeo_point_cartesian)
        nearest_point_cartesian = boundary.interpolate(projected)

        nearest_point_polar = (
            gpd.GeoSeries([nearest_point_cartesian], crs=3857)
            .to_crs(4326)
            .iloc[0]
        )

        p_closest = lps_dyn.Point.deg(nearest_point_polar.y, nearest_point_polar.x)
        dist_real = (point - p_closest).get_magnitude()

        if dist_real > tol:
            raise ValueError(
                f"Distance to nearest sediment polygon ({dist_real}) "
                f"exceeds tolerance ({tol})."
            )

        code = nearest["name"].strip()
        return SeabedProspector._convert_map[code], dist_real

class DepthProspector:
    """
    Query seafloor depth from an ETOPO 2022 dataset using xarray interpolation.

    The class keeps a static cache so the dataset is loaded only once,
    regardless of how many instances of DepthProspector are created.

    The ETOPO 2022 global relief model (NOAA) data is publicly available from:

        Metadata:
          https://www.ncei.noaa.gov/products/etopo-global-relief-model

        Direct download (30 Arc-Second Resolution, Bedrock elevation netCDF):
          https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/30s/30s_bed_elev_netcdf/catalog.html?dataset=globalDatasetScan/ETOPO2022/30s/30s_bed_elev_netcdf/ETOPO_2022_v1_30s_N90W180_bed.nc

    The dataset must contain a variable "z" representing elevation
        (positive above sea level, negative below sea level).
    """

    # Static cache
    _cached_ds = None
    _cached_z = None

    def __init__(self, etopo_file: str = "/data/ambiental/etopo/etopo_2022_30.nc"):
        """
        Load the ETOPO dataset only once and keep it cached.

        Parameters
        ----------
        etopo_file : str
            Path to the .nc ETOPO 2022 dataset.
        """

        if DepthProspector._cached_ds is None:
            ds = xr.open_dataset(etopo_file)

            if "z" not in ds:
                raise ValueError(
                    "Dataset must contain a variable named 'z' (ETOPO elevation)."
                )

            DepthProspector._cached_ds = ds
            DepthProspector._cached_z = ds["z"]

        self.ds = DepthProspector._cached_ds
        self.z = DepthProspector._cached_z

    def get(self, point: lps_dyn.Point) -> lps_qty.Distance:
        """
        Interpolate the ETOPO depth for a given position.

        Parameters
        ----------
        point : lps_dyn.Point

        Returns
        -------
        float
            Depth/elevation at the location.
            Positive values indicate ocean depth.
        """

        lat = point.latitude.get_deg()
        lon = point.longitude.get_deg()

        depth = self.z.interp(lat=lat, lon=lon, method="linear").item()
        if depth > 0:
            raise ValueError(
                f"{point}: ETOPO indicates land (positive elevation)."
            )

        return lps_qty.Distance.m(round(depth*10)/10 * -1)

class SSPProspector:
    """
    Query Sound Speed Profiles (SSP) from WOA18 seasonal CSV files.

    The class loads all salinity and temperature CSV files only once,
    storing them in static caches for fast repeated access.

    The World Ocean Atlas 2018 (WOA18) data is publicly available from:

        Metadata:
          https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/

        Direct download (temperature and salinity data in csv format and 1/4°):
            temperature data:
                https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA/temperature/csv/decav/0.25/woa18_t_decav_0.25_csv.tar.gz
          salinity data:
                https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA/salinity/csv/decav/0.25/woa18_s_decav_0.25_csv.tar.gz

    The csv file must be open and adjust the header:
        removing the comments in the begining
        adding the header in format [latitude, longitude, 0, 5, .... (other depths)]

    Expected file pattern (default):

    base_dir/
        winter_salinity.csv (file woa18_decav_s13mn04.csv)
        spring_salinity.csv (file woa18_decav_s14mn04.csv)
        summer_salinity.csv (file woa18_decav_s15mn04.csv)
        autumn_salinity.csv (file woa18_decav_s16mn04.csv)

        winter_temperature.csv (file woa18_decav_t13mn04.csv)
        spring_temperature.csv (file woa18_decav_t14mn04.csv)
        summer_temperature.csv (file woa18_decav_t15mn04.csv)
        autumn_temperature.csv (file woa18_decav_t16mn04.csv)
    """

    # Static cache
    _salinity_cache: typing.Dict[Season, pd.DataFrame] = {}
    _temperature_cache: typing.Dict[Season, pd.DataFrame] = {}

    # Standard WOA18 depth vector in meters
    _woa18_depths = np.array([
        0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,
        100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,
        550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,
        1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,
        1950,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,
        3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,
        4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500
    ])

    def __init__(self, base_dir: str = "/data/ambiental/woa18"):
        """
        Parameters
        ----------
        base_dir : str
            Directory containing WOA18 CSV files.
        """

        if not SSPProspector._salinity_cache:
            for season in Season:
                filename = os.path.join(base_dir, f"{season.name.lower()}_salinity.csv")
                SSPProspector._salinity_cache[season] = pd.read_csv(filename)

        if not SSPProspector._temperature_cache:
            for season in Season:
                filename = os.path.join(base_dir, f"{season.name.lower()}_temperature.csv")
                SSPProspector._temperature_cache[season] = pd.read_csv(filename)

        self.salinity_map = SSPProspector._salinity_cache
        self.temperature_map = SSPProspector._temperature_cache

    def get(self,
            point: lps_dyn.Point,
            season: Season,
            tol: lps_qty.Distance = lps_qty.Distance.km(30),
            max_depth: typing.Optional[lps_qty.Distance] = None):
        """
        Compute the Sound Speed Profile for given point and season.

        Parameters
        ----------
        point : lps_dyn.Point
        month : Season

        Returns
        -------
        depths : np.ndarray in meter
        sound_speed : np.ndarray in meter per seconds
        """

        lat = point.latitude.get_deg()
        lon = point.longitude.get_deg()

        df_s = self.salinity_map[season]
        df_t = self.temperature_map[season]

        # ---------------------------------------------------------------------
        # Find nearest salinity point
        df_s["dist"] = (df_s.iloc[:, 0] - lat)**2 + (df_s.iloc[:, 1] - lon)**2
        idx_s = df_s["dist"].idxmin()

        # Find nearest temperature point
        df_t["dist"] = (df_t.iloc[:, 0] - lat)**2 + (df_t.iloc[:, 1] - lon)**2
        idx_t = df_t["dist"].idxmin()

        row_s = df_s.loc[idx_s]
        row_t = df_t.loc[idx_t]

        p_sal = lps_dyn.Point.deg(row_s.iloc[0], row_s.iloc[1])
        dist_sal = (point - p_sal).get_magnitude()

        p_temp = lps_dyn.Point.deg(row_t.iloc[0], row_t.iloc[1])
        dist_temp = (point - p_temp).get_magnitude()

        if dist_sal > tol or dist_temp > tol:
            raise ValueError(
                f"For {point} there are no available data. "
                f"Distance to nearest SSP information ({dist_sal},{dist_temp}) "
                f"exceeds tolerance ({tol})."
            )

        # Extract profiles
        sal = row_s[2:-1].values.astype(float)
        tmp = row_t[2:-1].values.astype(float)

        # Same length
        n = min(len(sal), len(tmp))
        sal = sal[:n]
        tmp = tmp[:n]

        # Keep only where both valid
        both_mask = (~np.isnan(sal)) & (~np.isnan(tmp))
        sal = sal[both_mask]
        tmp = tmp[both_mask]

        # Depth vector matching the length
        depths = SSPProspector._woa18_depths[:n]
        depths = depths[both_mask]

        if max_depth is not None:
            current_max = depths[-1]
            max_depth = max_depth.get_m()

            if current_max < max_depth:

                depths = np.append(depths, max_depth)
                sal = np.append(sal, sal[-1])
                tmp = np.append(tmp, tmp[-1])

            elif current_max > max_depth:

                keep = depths <= max_depth
                if not np.any(keep):
                    raise ValueError("max_depth is shallower than first depth layer.")

                # Keep only allowed portion
                depths = depths[keep]
                sal = sal[keep]
                tmp = tmp[keep]

                # Ensure the depth exactly equals max_depth
                depths = np.append(depths, max_depth)
                sal = np.append(sal, sal[-1])
                tmp = np.append(tmp, tmp[-1])

        # Convert positive depth axis to negative z for GSW
        pres = gsw.p_from_z(-depths, lat)

        # Compute sound speed
        svp = gsw.sound_speed(sal, tmp, pres)

        depths = [lps_qty.Distance.m(d) for d in depths]
        svp = [lps_qty.Speed.m_s(s) for s in svp]
        return depths, svp

class EnvironmentProspector:
    """
    Query ECMWF Reanalysis v5 (ERA5) environmental data at a given location.

    The class loads the
        'Total precipitation' and
        'Significant height of combined wind waves and swell'
    from NetCDF4 (.nc) files only once, storing them in static caches for fast repeated access.

        Metadata:
            https://cds.climate.copernicus.eu/datasets

        Direct download:
            https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview

            Instructions:
                Monthly averaged reanalysis
                Total precipitation / Significant height of combined wind waves and swell
                2024
                all months
                NetCDF4

    """

    _cache_tp: typing.Optional[xr.Dataset] = None
    _cache_swh: typing.Optional[xr.Dataset] = None

    def __init__(
        self,
        tp_file: str = "/data/ambiental/era5/data_stream-moda_stepType-avgad.nc",
        swh_file: str = "/data/ambiental/era5/data_stream-wamd_stepType-avgua.nc",
    ):
        if EnvironmentProspector._cache_tp is None:
            EnvironmentProspector._cache_tp = xr.open_dataset(tp_file)

        if EnvironmentProspector._cache_swh is None:
            EnvironmentProspector._cache_swh = xr.open_dataset(swh_file)

        self.ds_tp : xr.Dataset = EnvironmentProspector._cache_tp
        self.ds_swh : xr.Dataset = EnvironmentProspector._cache_swh

    def _seasonal_values(self, da, season: Season, lat: float, lon: float) -> np.array:

        time_index = da["valid_time"].to_index()
        seasonal_mask = time_index.month.isin(season.get_months(lat))
        da_filtered = da.sel(valid_time=seasonal_mask)

        da_point = da_filtered.interp(latitude=lat, longitude=lon)

        return np.asarray(da_point.values, dtype=float)

    @staticmethod
    def _hs_to_douglas(hs: float) -> int:

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

    @staticmethod
    def _tp_to_seastate(tp: float) -> lps_env.Rain:
        boundaries = [
            (0, 0.004, lps_env.Rain.NONE),
            (0.004, 0.008, lps_env.Rain.LIGHT),
            (0.008, 0.016, lps_env.Rain.MODERATE),
            (0.016, 0.032, lps_env.Rain.HEAVY),
        ]

        for low, high, seastate in boundaries:
            if low <= tp < high:
                return seastate
        return lps_env.Rain.VERY_HEAVY

    def get_rain(self, point: lps_dyn.Point, season: Season) -> lps_env.Rain:
        """ Returns the highest average monthly value for a station converted by fixed values to
        the rain state. """

        lat = point.latitude.get_deg()
        lon = (360 + point.longitude.get_deg()) % 360

        tp = self._seasonal_values(
            self.ds_tp["tp"], season, lat, lon
        ).max()
        return EnvironmentProspector._tp_to_seastate(tp)

    def get_seastate(self, point: lps_dyn.Point, season: Season) -> lps_env.Sea:
        """ Returns the highest average monthly value for a station converted by Douglas Sea Scale.
        """

        lat = point.latitude.get_deg()
        lon = (360 + point.longitude.get_deg()) % 360

        hs = self._seasonal_values(
            self.ds_swh["swh"], season, lat, lon
        ).max()
        return EnvironmentProspector._hs_to_douglas(hs)

class AcousticSiteProspector:

    def __init__(self,
                 seabed_prospector = None,
                 depth_prospector = None,
                 ssp_prospector = None,
                 environment_prospector = None,):
        self.seabed_prospector = seabed_prospector or SeabedProspector()
        self.depth_prospector = depth_prospector or DepthProspector()
        self.ssp_prospector = ssp_prospector or SSPProspector()
        self.environment_prospector = environment_prospector or EnvironmentProspector()

    @staticmethod
    def _sensor_depth(depth: lps_qty.Distance) -> lps_qty.Distance:
        """ Return the local depth for the Location """
        if depth <= lps_qty.Distance.m(50):
            return depth - lps_qty.Distance.m(5)

        if depth < lps_qty.Distance.ft(600):
            return depth/2

        return lps_qty.Distance.m(100)

    def get_channel(self, point: lps_dyn.Point, season: Season, hash_id: str = None) -> \
            lps_channel.Channel:
        """ Return the lps_channel.Channel to the AcousticScenario. """

        max_depth = self.depth_prospector.get(point)
        seabed_type = self.seabed_prospector.get(point)

        depths, svp = self.ssp_prospector.get(
            point = point,
            season = season,
            max_depth = max_depth
        )

        desc = lps_desc.Description()
        for depth, speed in zip(depths, svp):
            desc.add(depth, speed)
        desc.add(max_depth, seabed_type.get_acoustical_layer())

        return lps_channel.Channel(description = desc,
                        sensor_depth = AcousticSiteProspector._sensor_depth(max_depth),
                        hash_id=hash_id)

    def get_env(self,
                point: lps_dyn.Point,
                season: Season,
                shipping_value: lps_env.Shipping = None,
                seed: int = None) -> lps_env.Environment:
        """ Return the lps_env.Environment to the AcousticScenario. """

        rain = self.environment_prospector.get_rain(point, season)
        seastate = self.environment_prospector.get_seastate(point, season)

        if shipping_value is None:
            rng = random.Random(seed)
            shipping_value = rng.uniform(lps_env.Shipping.NONE.value,
                                         lps_env.Shipping.LEVEL_7.value),

        return lps_env.Environment(
            rain_value = rain,
            sea_value = seastate,
            shipping_value = shipping_value,
            seed = seed
        )

    def get_complete_info(self, name: str, point: lps_dyn.Point) -> typing.Dict[str, str]:

        seabed, seabed_tol = self.seabed_prospector.get(point)
        depth = self.depth_prospector.get(point)

        ssp_max_depths = []
        rain_states = {}
        sea_states = {}
        for season in Season:
            dists, _ = self.ssp_prospector.get(point, season)
            rain_state = self.environment_prospector.get_rain(point, season)
            sea_state = self.environment_prospector.get_seastate(point, season)

            ssp_max_depths.append(dists[-1])
            rain_states[f"rain_state_{season}"] = rain_state
            sea_states[f"sea_state_{season}"] = sea_state

        return {
            "name": name,
            "latitude (deg)": point.latitude.get_deg(),
            "longitude (deg)": point.longitude.get_deg(),
            "latitude (dms)": str(point.latitude),
            "longitude (dms)": str(point.longitude),
            "seabed": seabed,
            "seabed_tol": seabed_tol,
            "local_depth": depth,
            "ssp_max_depth": max(ssp_max_depths),
            "ssp_max_diff": max(ssp_max_depths) - min(ssp_max_depths),
            "ssp_local_depth_margin": depth - max(ssp_max_depths),
            **rain_states,
            **sea_states,
        }

def _aligned_coords(min_v: float, max_v: float, fractions):
    """Generate aligned coordinates following .125 .375 .675 .875 rule."""
    vals = []
    base_min = math.floor(min_v)
    base_max = math.ceil(max_v) + 1
    for base in range(base_min - 1, base_max + 1):
        for f in fractions:
            v = base + f
            if (min_v - 1e-12) <= v <= (max_v + 1e-12):
                vals.append(round(v, 6))
    return sorted(set(vals))

def prospect_local(
    center_point: lps_dyn.Point,
    dist_lat: lps_qty.Distance,
    dist_lon: lps_qty.Distance,
    ssp: SSPProspector = None,
    seabed: SeabedProspector = None,
    depth_prosector: DepthProspector = None,
    desired_seabed: typing.Optional[syn_lay.SeabedType] = None,
    max_depth_dist: lps_qty.Distance = lps_qty.Distance.m(200),
):
    """
    Search for valid oceanographic points around a reference location using
    the SSPProspector and optional seabed constraints.

    Parameters
    ----------
    center_point : lps_dyn.Point
        Central geographic point from which the search window is defined.
    dist : lps_qty.Distance
        Half-size of the search window in both latitude and longitude.
    ssp : SSPProspector, optional
        SSP prospector used to retrieve sound speed profile data.
    seabed : SeabedProspector, optional
        Seabed prospector used to classify the seabed type.
    desired_seabed : syn_lay.SeabedType, optional
        If provided, points whose seabed type differs from this value
        are discarded.
    max_depth_dist : lps_qty.Distance, optional
        Maximum allowed difference between the deepest valid SSP point
        across all seasons. Points exceeding this threshold are discarded.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing all valid sampled points, with columns:
        'lat', 'lon', 'seabed_type', 'depths', 'max_depth', 'min_depth'.

    """
    ssp = ssp or SSPProspector()
    seabed = seabed or SeabedProspector()
    depth_prosector = depth_prosector or DepthProspector()

    disp = lps_dyn.Displacement(dist_lat, dist_lon)

    p_min = center_point - disp
    p_max = center_point + disp

    min_lat = p_min.latitude.get_deg()
    max_lat = p_max.latitude.get_deg()
    min_lon = p_min.longitude.get_deg()
    max_lon = p_max.longitude.get_deg()

    fractions = (0.125, 0.375, 0.675, 0.875)

    lats = _aligned_coords(min_lat, max_lat, fractions)
    lons = _aligned_coords(min_lon, max_lon, fractions)

    print(f"[INFO] Total latitudes: {len(lats)}, Total longitudes: {len(lons)}")

    results = []

    for lat in tqdm.tqdm(lats, desc="Latitudes", leave=False, ncols=120):
        for lon in tqdm.tqdm(lons, desc="Longitudes", leave=False, ncols=120):

            point = lps_dyn.Point.deg(lat, lon)

            seabed_type, seabed_error = None, None
            try:
                seabed_type, seabed_error = seabed.get(point)
            except ValueError:
                continue

            if desired_seabed is not None and seabed_type != desired_seabed:
                continue

            depths = []
            try:
                for season in Season:
                    depth, _ = ssp.get(point, season)

                    if len(depth) == 0:
                        continue

                    depths.append(depth[-1].get_m())
            except ValueError:
                continue

            if max(depths) - min(depths) > max_depth_dist.get_m():
                continue


            depth = None
            try:
                depth = depth_prosector.get(point)
            except ValueError:
                continue

            results.append({
                "lat": lat,
                "lon": lon,
                "seabed_type": seabed_type.name,
                "seabed_error": seabed_error.get_m(),
                "local_depth": depth.get_m(),
                "diff ssp depth": max(depths) - min(depths),
                "ssp_depths": depths,
                "depth_estimation_diff": max(depths) - depth.get_m()
            })

    df = pd.DataFrame(results)
    print(f"[INFO] total final de pontos válidos: {len(df)}")

    return df
