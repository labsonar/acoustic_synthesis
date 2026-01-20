"""
Scenario Module

Define AcousticScenario and its components.
"""
import enum
import random

import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_sce_dyn
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.environment.acoustic_site as lps_site
import lps_synthesis.propagation.channel_description as lps_desc
import lps_synthesis.propagation.channel as lps_channel
import lps_synthesis.propagation.layers as lps_layer

import lps_synthesis.database.catalog as syndb_core

class Location(enum.Enum):
    """ Enumeration of reference oceanic and coastal locations. """
    ADRIATIC_SEA = enum.auto()
    AMUNDSEN_SEA = enum.auto()
    ANDFJORDEN = enum.auto()
    ARGENTINE_SEA = enum.auto()
    BENGAL_BAY = enum.auto()
    EASTERN_SOUTH_PACIFIC_OCEAN = enum.auto()
    GUANABARA_BAY = enum.auto()
    GULF_OF_GUINEA = enum.auto()
    GULF_OF_ST_LAWRENCE = enum.auto()
    GULF_OF_THE_FARALLONES = enum.auto()
    MOZAMBIQUE_CHANNEL = enum.auto()
    NORTH_SEA = enum.auto()
    ONTONG_JAVA_PLATEAU = enum.auto()
    RIA_DE_VIGO = enum.auto()
    SANTOS_BASIN = enum.auto()
    STRAIT_OF_GEORGIA = enum.auto()
    STRAIT_OF_HORMUZ = enum.auto()
    TANPA_BAY = enum.auto()
    WESTERN_NORTH_PACIFIC_OCEAN = enum.auto()
    YUCATAN_BASIN = enum.auto()


    def get_point(self) -> lps_sce_dyn.Point:
        """ Returns the latitude and longitude of the selected location. """
        latlon_dict = {
            Location.ADRIATIC_SEA: [43.625, 14.375],
            Location.AMUNDSEN_SEA: [-72.125, -122.625],
            Location.ANDFJORDEN: [69.125, 16.375],
            Location.ARGENTINE_SEA: [-50.875, -67.625],
            Location.BENGAL_BAY: [15.625, 83.375],
            Location.EASTERN_SOUTH_PACIFIC_OCEAN: [-43.875, -90.125],
            Location.GUANABARA_BAY: [-23.125, -43.125],
            Location.GULF_OF_GUINEA: [3.125, 7.125],
            Location.GULF_OF_ST_LAWRENCE: [43.875, -60.875],
            Location.GULF_OF_THE_FARALLONES: [37.875, -122.875],
            Location.MOZAMBIQUE_CHANNEL: [-14.125, 45.875],
            Location.NORTH_SEA: [56.875, 2.625],
            Location.ONTONG_JAVA_PLATEAU: [-1.625, 158.675],
            Location.RIA_DE_VIGO: [42.125, -9.125],
            Location.SANTOS_BASIN: [-25.875, -42.625],
            Location.STRAIT_OF_GEORGIA: [49.125, -123.375],
            Location.STRAIT_OF_HORMUZ: [26.375, 56.625],
            Location.TANPA_BAY: [27.625, -82.875],
            Location.WESTERN_NORTH_PACIFIC_OCEAN: [30.125, 152.875],
            Location.YUCATAN_BASIN: [21.875, -85.125],

        }
        return lps_sce_dyn.Point.deg(*latlon_dict[self])

    def to_string(self, language: str = "en_US") -> str:
        """Returns the localized name of the location."""

        if language.lower() in ("pt_br", "pt-br"):
            names = {
                Location.ADRIATIC_SEA: "Mar Adriático",
                Location.AMUNDSEN_SEA: "Mar de Amundsen",
                Location.ANDFJORDEN: "Andfjorden",
                Location.ARGENTINE_SEA: "Mar da Argentina",
                Location.BENGAL_BAY: "Baía de Bengala",
                Location.EASTERN_SOUTH_PACIFIC_OCEAN: "Leste do Oceano Pacífico Sul",
                Location.GUANABARA_BAY: "Baía de Guanabara",
                Location.GULF_OF_GUINEA: "Golfo da Guiné",
                Location.GULF_OF_ST_LAWRENCE: "Golfo de São Lourenço",
                Location.GULF_OF_THE_FARALLONES: "Golfo dos Farallones",
                Location.MOZAMBIQUE_CHANNEL: "Canal de Moçambique",
                Location.NORTH_SEA: "Mar do Norte",
                Location.ONTONG_JAVA_PLATEAU: "Planalto Ontong Java",
                Location.RIA_DE_VIGO: "Ria de Vigo",
                Location.SANTOS_BASIN: "Bacia de Santos",
                Location.STRAIT_OF_GEORGIA: "Estreito de Geórgia",
                Location.STRAIT_OF_HORMUZ: "Estreito de Ormuz",
                Location.TANPA_BAY: "Baía de Tampa",
                Location.WESTERN_NORTH_PACIFIC_OCEAN: "Oeste do Oceano Pacífico Norte",
                Location.YUCATAN_BASIN: "Bacia de Yucatán",
            }
            return names[self]

        elif language.lower() in ("en_us", "en-us"):
            names = {
                Location.ADRIATIC_SEA: "Adriatic Sea",
                Location.AMUNDSEN_SEA: "Amundsen Sea",
                Location.ANDFJORDEN: "Andfjorden",
                Location.ARGENTINE_SEA: "Argentine Sea",
                Location.BENGAL_BAY: "Bay of Bengal",
                Location.EASTERN_SOUTH_PACIFIC_OCEAN: "Eastern South Pacific Ocean",
                Location.GUANABARA_BAY: "Guanabara Bay",
                Location.GULF_OF_GUINEA: "Gulf of Guinea",
                Location.GULF_OF_ST_LAWRENCE: "Gulf of St. Lawrence",
                Location.GULF_OF_THE_FARALLONES: "Gulf of the Farallones",
                Location.MOZAMBIQUE_CHANNEL: "Mozambique Channel",
                Location.NORTH_SEA: "North Sea",
                Location.ONTONG_JAVA_PLATEAU: "Ontong Java Plateau",
                Location.RIA_DE_VIGO: "Ria of Vigo",
                Location.SANTOS_BASIN: "Santos Basin",
                Location.STRAIT_OF_GEORGIA: "Strait of Georgia",
                Location.STRAIT_OF_HORMUZ: "Strait of Hormuz",
                Location.TANPA_BAY: "Tampa Bay",
                Location.WESTERN_NORTH_PACIFIC_OCEAN: "Western North Pacific Ocean",
                Location.YUCATAN_BASIN: "Yucatan Basin",
            }
            return names[self]

    def get_shipping(self) -> lps_env.Shipping:
        """Return the typical shipping level for this Location.

          https://www.arcgis.com/apps/mapviewer/index.html?layers=2f72eb72cc0b403bb19a7cd1853f3d94
        """
        shipping_map = {
            Location.ADRIATIC_SEA: lps_env.Shipping.LEVEL_6,
            Location.AMUNDSEN_SEA: lps_env.Shipping.NONE,
            Location.ANDFJORDEN: lps_env.Shipping.LEVEL_5,
            Location.ARGENTINE_SEA: lps_env.Shipping.LEVEL_1,
            Location.BENGAL_BAY: lps_env.Shipping.LEVEL_2,
            Location.EASTERN_SOUTH_PACIFIC_OCEAN: lps_env.Shipping.LEVEL_1,
            Location.GUANABARA_BAY: lps_env.Shipping.LEVEL_3,
            Location.GULF_OF_GUINEA: lps_env.Shipping.LEVEL_4,
            Location.GULF_OF_ST_LAWRENCE: lps_env.Shipping.LEVEL_4,
            Location.GULF_OF_THE_FARALLONES: lps_env.Shipping.LEVEL_6,
            Location.MOZAMBIQUE_CHANNEL: lps_env.Shipping.LEVEL_2,
            Location.NORTH_SEA: lps_env.Shipping.LEVEL_5,
            Location.ONTONG_JAVA_PLATEAU: lps_env.Shipping.LEVEL_2,
            Location.RIA_DE_VIGO: lps_env.Shipping.LEVEL_7,
            Location.SANTOS_BASIN: lps_env.Shipping.LEVEL_3,
            Location.STRAIT_OF_GEORGIA: lps_env.Shipping.LEVEL_7,
            Location.STRAIT_OF_HORMUZ: lps_env.Shipping.LEVEL_6,
            Location.TANPA_BAY: lps_env.Shipping.LEVEL_6,
            Location.WESTERN_NORTH_PACIFIC_OCEAN: lps_env.Shipping.LEVEL_3,
            Location.YUCATAN_BASIN: lps_env.Shipping.LEVEL_4,
        }

        return shipping_map[self]

    def __str__(self):
        return self.to_string()

    def as_dict(self):
        """ Return Local as dict to save local description. """
        p = self.get_point()
        return {
                "Local ID": self.value,
                "Name (us)": self.to_string(),
                "Latitude (deg)": p.latitude.get_deg(),
                "Longitude (deg)": p.longitude.get_deg(),
                "Latitude (dms)": str(p.latitude),
                "Longitude (dms)": str(p.longitude),
            }

    @staticmethod
    def plot(filename: str):
        """ Plot all defined locations on a world map. """

        plt.figure(figsize=(16, 9), dpi=600)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(resolution='110m', linewidth=0.6)
        ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')

        for local in Location:
            p = local.get_point()
            color='blue' # if local.is_shallow_water() else 'red'
            offset=1.5 # if local.is_shallow_water() else -4.5
            ax.plot(p.longitude.get_deg(), p.latitude.get_deg(), 'o',
                    markersize=4, color=color, transform=ccrs.PlateCarree())
            ax.text(p.longitude.get_deg() + offset, p.latitude.get_deg(), local.value,
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
                 local: Location = None,
                 season: lps_site.Season = None,
                 prospector: lps_site.AcousticSiteProspector = None):
        self.local = local or Location.rand()
        self.season = season or lps_site.Season.rand()
        self.prospector = prospector or lps_site.AcousticSiteProspector()

    def __str__(self):
        return f"{self.local} [{self.season.name.capitalize()}]"

    def as_dict(self):
        """Return the scenario as a dictionary joining Local and Month information."""
        return {
            **self.local.as_dict(),
            "Season": self.season.name.capitalize(),
        }

    def get_env(self, seed: int = None) -> lps_env.Environment:
        """ Return the lps_env.Environment to the AcousticScenario. """
        return self.prospector.get_env(point = self.local.get_point(),
                                       season = self.season,
                                       shipping_value = self.local.get_shipping(),
                                       seed = seed)

    def get_channel(self) -> lps_channel.Channel:
        """ Return the lps_channel.Channel to the AcousticScenario. """
        return self.prospector.get_channel(point = self.local.get_point(),
                                       season = self.season,
                                       hash_id=self.local.name.lower())
