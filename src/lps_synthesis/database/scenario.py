"""
Scenario Module

Define AcousticScenario and its components.
"""
import os
import enum
import random
import typing

import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes as cgeo

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_sce_dyn
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.environment.acoustic_site as lps_site
import lps_synthesis.propagation.channel as lps_channel
import lps_synthesis.propagation.channel_description as lps_channel_desc
import lps_synthesis.propagation.channel_response as lps_channel_rsp
import lps_synthesis.propagation.models as lps_propag_model

import lps_synthesis.database.catalog as syndb_core

class Location(enum.Enum):
    """ Enumeration of reference oceanic and coastal locations. """
    ARABIAN_SEA = enum.auto()
    ARGENTINE_SEA = enum.auto()
    BALTIC_SEA = enum.auto()
    BAY_OF_BENGAL = enum.auto()
    CORAL_SEA = enum.auto()
    EAST_CHINA_SEA = enum.auto()
    GUANABARA_BAY = enum.auto()
    GULF_OF_ADEN = enum.auto()
    GULF_OF_BOTHNIA = enum.auto()
    GULF_OF_CADIZ = enum.auto()
    GULF_OF_GUINEA = enum.auto()
    HEBRIDES_SEA = enum.auto()
    QUEEN_CHARLOTTE_SOUND = enum.auto()
    RIO_DE_LA_PLATA_ESTUARY = enum.auto()
    SCOTIA_SEA = enum.auto()
    SEA_OF_JAPAN = enum.auto()
    SOUTH_CHINA_SEA = enum.auto()
    TASMAN_SEA = enum.auto()
    WESTERN_SOUTH_PACIFIC_OCEAN = enum.auto()
    YUCATAN_CHANNEL = enum.auto()

    def get_point(self) -> lps_sce_dyn.Point:
        """ Returns the latitude and longitude of the selected location. """
        latlon_dict = {
            Location.ARABIAN_SEA: [23.375, 67.875],
            Location.ARGENTINE_SEA: [-50.875, -67.625],
            Location.BALTIC_SEA: [55.125, 12.875],
            Location.BAY_OF_BENGAL: [18.875, 89.375],
            Location.CORAL_SEA: [-21.875, 151.125],
            Location.EAST_CHINA_SEA: [25.125, 122.125],
            Location.GUANABARA_BAY: [-23.125, -43.125],
            Location.GULF_OF_ADEN: [12.375, 51.125],
            Location.GULF_OF_BOTHNIA: [64.675, 23.875],
            Location.GULF_OF_CADIZ: [36.875, -8.325],
            Location.GULF_OF_GUINEA: [3.875, 7.375],
            Location.HEBRIDES_SEA: [56.875, -7.625],
            Location.QUEEN_CHARLOTTE_SOUND: [51.375, -129.125],
            Location.RIO_DE_LA_PLATA_ESTUARY: [-36.325, -55.875],
            Location.SCOTIA_SEA: [-53.625, -38.875],
            Location.SEA_OF_JAPAN: [43.875, 141.125],
            Location.SOUTH_CHINA_SEA: [10.675, 117.675],
            Location.TASMAN_SEA: [-39.875, 168.125],
            Location.WESTERN_SOUTH_PACIFIC_OCEAN: [-12.125, 13.125],
            Location.YUCATAN_CHANNEL: [21.875, -85.125],
        }
        return lps_sce_dyn.Point.deg(*latlon_dict[self])

    def to_string(self, language: str = "en_US") -> str:
        """Return the localized name of the location."""

        names_pt_br = {
            Location.ARABIAN_SEA: "Mar da Arábia",
            Location.ARGENTINE_SEA: "Mar da Argentina",
            Location.BALTIC_SEA: "Mar Báltico",
            Location.BAY_OF_BENGAL: "Baía de Bengala",
            Location.CORAL_SEA: "Mar de Coral",
            Location.EAST_CHINA_SEA: "Mar da China Oriental",
            Location.GULF_OF_ADEN: "Golfo de Áden",
            Location.GULF_OF_BOTHNIA: "Golfo de Bótnia",
            Location.GULF_OF_CADIZ: "Golfo de Cádis",
            Location.GULF_OF_GUINEA: "Golfo da Guiné",
            Location.GUANABARA_BAY: "Baía de Guanabara",
            Location.HEBRIDES_SEA: "Mar das Hébridas",
            Location.QUEEN_CHARLOTTE_SOUND: "Queen Charlotte Sound",
            Location.RIO_DE_LA_PLATA_ESTUARY: "Estuário do Rio da Prata",
            Location.SCOTIA_SEA: "Mar de Scotia",
            Location.SEA_OF_JAPAN: "Mar do Japão",
            Location.SOUTH_CHINA_SEA: "Mar do Sul da China",
            Location.TASMAN_SEA: "Mar da Tasmânia",
            Location.WESTERN_SOUTH_PACIFIC_OCEAN: "Oeste do Oceano Pacífico Sul",
            Location.YUCATAN_CHANNEL: "Canal de Yucatán",
        }

        names_en_us = {
            Location.ARABIAN_SEA: "Arabian Sea",
            Location.ARGENTINE_SEA: "Argentine Sea",
            Location.BALTIC_SEA: "Baltic Sea",
            Location.BAY_OF_BENGAL: "Bay of Bengal",
            Location.CORAL_SEA: "Coral Sea",
            Location.EAST_CHINA_SEA: "East China Sea",
            Location.GULF_OF_ADEN: "Gulf of Aden",
            Location.GULF_OF_BOTHNIA: "Gulf of Bothnia",
            Location.GULF_OF_CADIZ: "Gulf of Cadiz",
            Location.GULF_OF_GUINEA: "Gulf of Guinea",
            Location.GUANABARA_BAY: "Guanabara Bay",
            Location.HEBRIDES_SEA: "Hebrides Sea",
            Location.QUEEN_CHARLOTTE_SOUND: "Queen Charlotte Sound",
            Location.RIO_DE_LA_PLATA_ESTUARY: "Rio de la Plata Estuary",
            Location.SCOTIA_SEA: "Scotia Sea",
            Location.SEA_OF_JAPAN: "Sea of Japan",
            Location.SOUTH_CHINA_SEA: "South China Sea",
            Location.TASMAN_SEA: "Tasman Sea",
            Location.WESTERN_SOUTH_PACIFIC_OCEAN: "Western South Pacific Ocean",
            Location.YUCATAN_CHANNEL: "Yucatan Channel",
        }

        if language.lower() in ("pt_br", "pt-br"):
            return names_pt_br[self]

        if language.lower() in ("en_us", "en-us"):
            return names_en_us[self]

        raise NotImplementedError(f"Location.to_string not implemented for language: {language}")

    def get_shipping_level(self) -> lps_env.Shipping:
        """Return the typical shipping level for this Location.

          https://www.arcgis.com/apps/mapviewer/index.html?layers=2f72eb72cc0b403bb19a7cd1853f3d94
        """
        shipping_map = {
            Location.ARABIAN_SEA: lps_env.Shipping.LEVEL_4,
            Location.ARGENTINE_SEA: lps_env.Shipping.LEVEL_1,
            Location.BALTIC_SEA: lps_env.Shipping.LEVEL_7,
            Location.BAY_OF_BENGAL: lps_env.Shipping.LEVEL_3,
            Location.CORAL_SEA: lps_env.Shipping.LEVEL_2,
            Location.EAST_CHINA_SEA: lps_env.Shipping.LEVEL_5,
            Location.GUANABARA_BAY: lps_env.Shipping.LEVEL_3,
            Location.GULF_OF_ADEN: lps_env.Shipping.LEVEL_4,
            Location.GULF_OF_BOTHNIA: lps_env.Shipping.LEVEL_6,
            Location.GULF_OF_CADIZ: lps_env.Shipping.LEVEL_6,
            Location.GULF_OF_GUINEA: lps_env.Shipping.LEVEL_4,
            Location.HEBRIDES_SEA: lps_env.Shipping.LEVEL_6,
            Location.QUEEN_CHARLOTTE_SOUND: lps_env.Shipping.LEVEL_2,
            Location.RIO_DE_LA_PLATA_ESTUARY: lps_env.Shipping.LEVEL_2,
            Location.SCOTIA_SEA: lps_env.Shipping.LEVEL_1,
            Location.SEA_OF_JAPAN: lps_env.Shipping.LEVEL_5,
            Location.SOUTH_CHINA_SEA: lps_env.Shipping.LEVEL_3,
            Location.TASMAN_SEA: lps_env.Shipping.LEVEL_1,
            Location.WESTERN_SOUTH_PACIFIC_OCEAN: lps_env.Shipping.LEVEL_2,
            Location.YUCATAN_CHANNEL: lps_env.Shipping.LEVEL_4,
        }

        return shipping_map[self]

    def __repr__(self) -> str:
        return self.to_string()

    def as_dict(self):
        """ Return Local as dict to save local description. """
        p = self.get_point()
        return {
                "LOCAL_ID": self.value,
                "NAME_(US)": self.to_string(),
                "LATITUDE_(DEG)": p.latitude.get_deg(),
                "LONGITUDE_(DEG)": p.longitude.get_deg(),
                "LATITUDE_(DMS)": str(p.latitude),
                "LONGITUDE_(DMS)": str(p.longitude),
            }

    def is_shallow_water(self, etopo_file: str | None = None) -> bool:
        """ Indicates whether the location is classified as shallow water. """
        if etopo_file is None:
            prospector = lps_site.DepthProspector()
        else:
            prospector = lps_site.DepthProspector(etopo_file=etopo_file)

        return prospector.get(self.get_point()) < lps_qty.Distance.ft(600)
        # deep water (> 600 ft)
        # R. P. Hodges, Underwater acoustics: analysis, design, and performance of sonar.
        # Hoboken, NJ: Wiley, 2010. doi: 10.1002/9780470665244.

    @staticmethod
    def plot(filename: str):
        """ Plot all defined locations on a world map. """

        plt.figure(figsize=(16, 9), dpi=600)
        ax = typing.cast(cgeo.GeoAxes, plt.axes(projection=ccrs.PlateCarree()))
        ax.set_global()
        ax.coastlines(resolution='110m', linewidth=0.6)
        ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')

        for local in Location:
            p = local.get_point()
            color='blue' if local.is_shallow_water() else 'red'
            offset=1.5 if local.is_shallow_water() else -4.5
            ax.plot(p.longitude.get_deg(), p.latitude.get_deg(), 'o',
                    markersize=4, color=color, transform=ccrs.PlateCarree())
            ax.text(p.longitude.get_deg() + offset, p.latitude.get_deg(), f"{local.value}",
                    fontsize=6, color=color, transform=ccrs.PlateCarree())

        plt.savefig(filename, dpi=600, bbox_inches="tight")

    @staticmethod
    def to_df():
        """Save all locations as a DataFrame and export to CSV."""
        data = []
        for local in Location:
            data.append(local.as_dict())

        df = pd.DataFrame(data)
        return df

    @staticmethod
    def rand() -> "Location":
        """Return a random Local."""
        return random.choice(list(Location))


class AcousticScenario(syndb_core.CatalogEntry):
    """ Class to represent an Acoustic Scenario"""

    def __init__(self,
                 local: Location | None = None,
                 season: lps_site.Season | None = None,
                 prospector: lps_site.AcousticSiteProspector | None = None):
        self.local = local or Location.rand()
        self.season = season or lps_site.Season.rand()
        self._prospector = prospector
        self._df = None

        csv_path = os.path.join(os.path.dirname(__file__), "data", "acoustic_scenario_info.csv")
        if os.path.exists(csv_path):
            self._df = pd.read_csv(csv_path)

    def __str__(self):
        return f"{self.local} [{self.season.name.capitalize()}]"

    def as_dict(self):
        """Return the scenario as a dictionary joining Local and Month information."""
        return {
            **self.local.as_dict(),
            "Season": self.season.name.capitalize(),
        }

    def get_prospector(self) -> lps_site.AcousticSiteProspector:
        if self._prospector is None:
            self._prospector = lps_site.AcousticSiteProspector()
        return self._prospector

    def _query_df(self, model_name: str | None):
        if self._df is None:
            raise KeyError("Acoustic Scenario info not found")

        if model_name is None:
            row = self._df.query(
                "LOCAL == @self.local.name and "
                "SEASON == @self.season.name"
            )
        else:
            row = self._df.query(
                "LOCAL == @self.local.name and "
                "SEASON == @self.season.name and "
                "MODEL == @model_name"
            )

        if row.empty:
            raise KeyError(
                f"Scenario not found in CSV: "
                f"{self.local.name} / {self.season.name} / {model_name}"
            )

        return row.iloc[0]


    def get_env(self, seed: int | None = None) -> lps_env.Environment:
        """ Return the lps_env.Environment to the AcousticScenario. """
        try:
            row = self._query_df(model_name=None)

            return lps_env.Environment(
                rain_value=row["RAIN_VALUE"],
                sea_value=row["SEA_VALUE"],
                shipping_value=row["SHIPPING_VALUE"],
                seed=int(row["SEED"]),
            )
        except KeyError:
            pass

        return self.get_prospector().get_env(point = self.local.get_point(),
                                       season = self.season,
                                       shipping_value = self.local.get_shipping_level(),
                                       seed = seed)

    def get_channel(self,
                    model: lps_propag_model.PropagationModel | None = None) -> lps_channel.Channel:
        """ Return the lps_channel.Channel to the AcousticScenario. """

        hash_id=f"{self.local.name.lower()}"
        # hash_id=f"{self.local.name.lower()}_{self.season.name.lower()}"

        if self._df is not None:
            row = self._query_df(model_name=str(model) if model is not None else None)

            desc_filename = f'{row["LOCAL"].lower()}_{row["SEASON"].lower()}.pkl'
            desc_filename = os.path.join(lps_channel.DEFAULT_DIR, desc_filename)

            if os.path.exists(desc_filename):

                sensor_depth = lps_qty.Distance.m(row["SENSOR_DEPTH"])
                desc = lps_channel_desc.Description.load(desc_filename)

                query = lps_site.AcousticSiteProspector.get_default_query(desc, sensor_depth)

                channel_filname = os.path.join(lps_channel.DEFAULT_DIR, row["CHANNEL_FILENAME"])

                response = lps_channel_rsp.TemporalResponse.load(channel_filname)

                ch = lps_channel.Channel.__new__(lps_channel.Channel)

                ch.query = query
                ch.model = lps_propag_model.Type[row["MODEL"].upper()].build_model()
                ch.hash_id = hash_id
                ch.response = response
                ch.channel_dir = lps_channel.DEFAULT_DIR

                return ch


        return self.get_prospector().get_channel(point = self.local.get_point(),
                                       season = self.season,
                                       model = model,
                                       hash_id=hash_id)

    @classmethod
    def load_catalog(cls, filename: str) -> syndb_core.Catalog["AcousticScenario"]:
        df = pd.read_csv(filename)
        scenarios = []

        for _, row in df.iterrows():
            local = Location(row["LOCAL_ID"])
            season = lps_site.Season[row["Season"].upper()]

            scenarios.append(
                AcousticScenario(
                    local=local,
                    season=season
                )
            )

        return syndb_core.Catalog[AcousticScenario](entries=scenarios)
