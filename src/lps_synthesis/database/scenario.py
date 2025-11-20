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

import lps_synthesis.scenario.dynamic as lps_sce_dyn
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.channel as lps_channel

import lps_synthesis.database.catalog as syndb_core

class Local(enum.Enum):
    """ Enumeration of reference oceanic and coastal locations. """

    ADRIATIC_SEA = enum.auto()
    ANDFJORDEN = enum.auto()
    ANTIGUA_BARBUDA = enum.auto()
    BENGAL_BAY = enum.auto()
    CARIBBEAN_SEA = enum.auto()
    NORTH_SEA = enum.auto()
    GUANABARA_BAY = enum.auto()
    GULF_OF_PENAS = enum.auto()
    GULF_OF_GUINEA = enum.auto()
    GULF_OF_THE_FARALLONES = enum.auto()
    NARINDA_BAY = enum.auto()
    WESTERN_PACIFIC_OCEAN_NORTH = enum.auto()
    ONTONG_JAVA_PLATEAU = enum.auto()
    AMUNDSEN_SEA = enum.auto()
    SANTOS_BASIN = enum.auto()
    EASTERN_PACIFIC_OCEAN_SOUTH = enum.auto()
    STRAIT_OF_GEORGIA = enum.auto()
    STRAIT_OF_HORMUZ = enum.auto()
    TANPA_BAY = enum.auto()
    RIA_DE_VIGO = enum.auto()


    def get_point(self) -> lps_sce_dyn.Point:
        """ Returns the latitude and longitude of the selected location. """
        latlon_dict = {
            Local.ADRIATIC_SEA: [43.625, 14.375],
            Local.ANDFJORDEN: [69.125, 16.375],
            Local.ANTIGUA_BARBUDA: [17.375, -61.625],
            Local.BENGAL_BAY: [16.625, 84.375],
            Local.CARIBBEAN_SEA: [12.125, -81.875],
            Local.NORTH_SEA: [56.875, 2.625],
            Local.GUANABARA_BAY: [-23.125, -43.125],
            Local.GULF_OF_PENAS: [-46.875, -75.875],
            Local.GULF_OF_GUINEA: [0.125, 3.875],
            Local.GULF_OF_THE_FARALLONES: [37.875, -122.875],
            Local.NARINDA_BAY: [-14.625, 47.625],
            Local.WESTERN_PACIFIC_OCEAN_NORTH: [33.625, 150.125],
            Local.ONTONG_JAVA_PLATEAU: [-7.875, 159.625],
            Local.AMUNDSEN_SEA: [-70.625, -95.875],
            Local.SANTOS_BASIN: [-25.125, -43.125],
            Local.EASTERN_PACIFIC_OCEAN_SOUTH: [-43.625, -89.875],
            Local.STRAIT_OF_GEORGIA: [49.125, -123.375],
            Local.STRAIT_OF_HORMUZ: [26.375, 56.625],
            Local.TANPA_BAY: [27.625, -82.875],
            Local.RIA_DE_VIGO: [42.125, -9.125],
        }
        return lps_sce_dyn.Point.deg(*latlon_dict[self])

    def to_string(self, language: str = "en_US") -> str:
        """ Returns the localized name of the location. """

        if language == "pt_br":
            names = {
                Local.GUANABARA_BAY: "Baía de Guanabara",
                Local.VIGO_PORT: "Porto de Vigo",
                Local.QIANDAO_LAKE: "Lago Qiandao",
                Local.DOGGER_BANK: "Banco Dogger",
                Local.CAMPECHE_BAY: "Baía de Campeche",
                Local.TOKYO_BAY: "Baía de Tóquio",
                Local.STRAIT_OF_HORMUZ: "Estreito de Ormuz",
                Local.EXMOUTH_GULF: "Golfo de Exmouth",
                Local.GULF_OF_GUINEA: "Golfo da Guiné",
                Local.GULF_OF_THE_FARALLONES: "Golfo dos Faralhões",
                Local.SANTOS_BASIN: "Bacia de Santos",
                Local.STRAIT_OF_GEORGIA: "Estreito de Geórgia",
                Local.BERMUDA_RISE: "Elevação das Bermudas",
                Local.WALVIS_RIDGE: "Dorsal de Walvis",
                Local.MARIANA_BASIN: "Bacia das Marianas",
                Local.HAWAII_RIDGE: "Dorsal do Havaí",
                Local.DRAKE_PASSAGE: "Passagem de Drake",
                Local.ARABIAN_SEA: "Mar Arábico",
                Local.TASMAN_SEA: "Mar da Tasmânia",
                Local.BARENTS_SEA: "Mar de Barents",
            }
            return names[self]

        elif language == "en_US":
            names = {
                Local.GUANABARA_BAY: "Guanabara Bay",
                Local.VIGO_PORT: "Vigo Port",
                Local.QIANDAO_LAKE: "Qiandao Lake",
                Local.DOGGER_BANK: "Dogger Bank",
                Local.CAMPECHE_BAY: "Campeche Bay",
                Local.TOKYO_BAY: "Tokyo Bay",
                Local.STRAIT_OF_HORMUZ: "Strait of Hormuz",
                Local.EXMOUTH_GULF: "Exmouth Gulf",
                Local.GULF_OF_GUINEA: "Gulf of Guinea",
                Local.GULF_OF_THE_FARALLONES: "Gulf of the Farallones",
                Local.SANTOS_BASIN: "Santos Basin",
                Local.STRAIT_OF_GEORGIA: "Strait of Georgia",
                Local.BERMUDA_RISE: "Bermuda Rise",
                Local.WALVIS_RIDGE: "Walvis Ridge",
                Local.MARIANA_BASIN: "Mariana Basin",
                Local.HAWAII_RIDGE: "Hawaii Ridge",
                Local.DRAKE_PASSAGE: "Drake Passage",
                Local.ARABIAN_SEA: "Arabian Sea",
                Local.TASMAN_SEA: "Tasman Sea",
                Local.BARENTS_SEA: "Barents Sea",
            }
            return names[self]
        else:
            raise NotImplementedError(f"Localization not available for language '{language}'.")

    def is_shallow_water(self) -> bool:
        """ Indicates whether the location is classified as shallow water. """
        shallow_sites = {
            Local.ADRIATIC_SEA,
            Local.NORTH_SEA,
            Local.GUANABARA_BAY,
            Local.GULF_OF_THE_FARALLONES,
            Local.NARINDA_BAY,
            Local.STRAIT_OF_HORMUZ,
            Local.TANPA_BAY,
            Local.RIA_DE_VIGO,
        }
        return self in shallow_sites

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

        for local in Local:
            p = local.get_point()
            color='blue' if local.is_shallow_water() else 'red'
            offset=1.5 if local.is_shallow_water() else -4.5
            ax.plot(p.longitude.get_deg(), p.latitude.get_deg(), 'o',
                    markersize=4, color=color, transform=ccrs.PlateCarree())
            ax.text(p.longitude.get_deg() + offset, p.latitude.get_deg(), local.value,
                    fontsize=6, color=color, transform=ccrs.PlateCarree())

        plt.savefig(filename, dpi=600, bbox_inches="tight")

    @staticmethod
    def to_df():
        """Save all locations as a DataFrame and export to CSV."""
        data = []
        for local in Local:
            data.append(local.as_dict())

        df = pd.DataFrame(data)
        return df

    @staticmethod
    def rand() -> "Local":
        """Return a random Local."""
        return random.choice(list(Local))

class Month(enum.IntEnum):
    """ Enum to represent month. """
    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12

    @staticmethod
    def rand() -> "Month":
        """Return a random month."""
        return random.choice(list(Month))

class AcousticScenario(syndb_core.CatalogEntry):
    """ Class to represent an Acoustic Scenario"""

    def __init__(self, local: Local = None, month: Month = None):
        self.local = local or Local.rand()
        self.month = month or Month.rand()

    def __str__(self):
        return f"{self.local} [{self.month.name.capitalize()}]"

    def as_dict(self):
        """Return the scenario as a dictionary joining Local and Month information."""
        return {
            **self.local.as_dict(),
            "Month": self.month.name.capitalize(),
        }

    def get_env(self) -> lps_env.Environment:
        """ Return the lps_env.Environment to the AcousticScenario. """
        raise NotImplementedError("AcousticScenario.get_env")

    def get_channel(self) -> lps_channel.Channel:
        """ Return the lps_channel.Channel to the AcousticScenario. """
        raise NotImplementedError("AcousticScenario.get_channel")
