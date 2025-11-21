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
import lps_synthesis.propagation.layers as lps_layer

import lps_synthesis.database.catalog as syndb_core

class Location(enum.Enum):
    """ Enumeration of reference oceanic and coastal locations. """

    ADRIATIC_SEA = enum.auto()
    AMUNDSEN_SEA = enum.auto()
    ANDFJORDEN = enum.auto()
    BENGAL_BAY = enum.auto()
    EASTERN_PACIFIC_OCEAN_SOUTH = enum.auto()
    GUANABARA_BAY = enum.auto()
    GULF_OF_GUINEA = enum.auto()
    GULF_OF_PENAS = enum.auto()
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
    WESTERN_PACIFIC_OCEAN_NORTH = enum.auto()
    YUCATAN_BASIN = enum.auto()



    def get_point(self) -> lps_sce_dyn.Point:
        """ Returns the latitude and longitude of the selected location. """
        latlon_dict = {
            Location.ADRIATIC_SEA: [43.625, 14.375],
            Location.AMUNDSEN_SEA: [-73.875, -107.325],
            Location.ANDFJORDEN: [69.125, 16.375],
            Location.BENGAL_BAY: [15.625, 83.375],
            Location.EASTERN_PACIFIC_OCEAN_SOUTH: [-43.625, -89.875],
            Location.GUANABARA_BAY: [-23.125, -43.125],
            Location.GULF_OF_GUINEA: [3.125, 7.125],
            Location.GULF_OF_PENAS: [-47.625, -75.325],
            Location.GULF_OF_ST_LAWRENCE: [43.875, -60.875],
            Location.GULF_OF_THE_FARALLONES: [37.875, -122.875],
            Location.MOZAMBIQUE_CHANNEL: [-15.625, 45.675],
            Location.NORTH_SEA: [56.875, 2.625],
            Location.ONTONG_JAVA_PLATEAU: [-1.625, 158.675],
            Location.RIA_DE_VIGO: [42.125, -9.125],
            Location.SANTOS_BASIN: [-25.875, -42.625],
            Location.STRAIT_OF_GEORGIA: [49.125, -123.375],
            Location.STRAIT_OF_HORMUZ: [26.375, 56.625],
            Location.TANPA_BAY: [27.625, -82.875],
            Location.WESTERN_PACIFIC_OCEAN_NORTH: [30.125, 152.875],
            Location.YUCATAN_BASIN: [21.875, -85.125],

        }
        return lps_sce_dyn.Point.deg(*latlon_dict[self])

    def to_string(self, language: str = "en_US") -> str:
        """Returns the localized name of the location."""
        return self.name.replace("_", " ").capitalize()

        # if language.lower() in ("pt_br", "pt-br"):
        #     names = {
        #         Location.ADRIATIC_SEA: "Mar Adriático",
        #         Location.ANDFJORDEN: "Andfjorden",
        #         Location.ANTIGUA_BARBUDA: "Antígua e Barbuda",
        #         Location.BENGAL_BAY: "Baía de Bengala",
        #         Location.CARIBBEAN_SEA: "Mar do Caribe",
        #         Location.NORTH_SEA: "Mar do Norte",
        #         Location.GUANABARA_BAY: "Baía de Guanabara",
        #         Location.GULF_OF_PENAS: "Golfo de Penas",
        #         Location.GULF_OF_GUINEA: "Golfo da Guiné",
        #         Location.GULF_OF_THE_FARALLONES: "Golfo dos Farallones",
        #         Location.NARINDA_BAY: "Baía de Narinda",
        #         Location.WESTERN_PACIFIC_OCEAN_NORTH: "Leste do Oceano Pacífico Norte",
        #         Location.ONTONG_JAVA_PLATEAU: "Planalto Ontong Java",
        #         Location.AMUNDSEN_SEA: "Mar de Amundsen",
        #         Location.SANTOS_BASIN: "Bacia de Santos",
        #         Location.EASTERN_PACIFIC_OCEAN_SOUTH: "Oeste do Oceano Pacífico Norte",
        #         Location.STRAIT_OF_GEORGIA: "Estreito de Geórgia",
        #         Location.STRAIT_OF_HORMUZ: "Estreito de Ormuz",
        #         Location.TANPA_BAY: "Baía de Tampa",
        #         Location.RIA_DE_VIGO: "Ria de Vigo",
        #     }
        #     return names[self]

        # elif language.lower() in ("en_us", "en-us"):
        #     names = {
        #         Location.ADRIATIC_SEA: "Adriatic Sea",
        #         Location.ANDFJORDEN: "Andfjorden",
        #         Location.ANTIGUA_BARBUDA: "Antigua and Barbuda",
        #         Location.BENGAL_BAY: "Bay of Bengal",
        #         Location.CARIBBEAN_SEA: "Caribbean Sea",
        #         Location.NORTH_SEA: "North Sea",
        #         Location.GUANABARA_BAY: "Guanabara Bay",
        #         Location.GULF_OF_PENAS: "Gulf of Penas",
        #         Location.GULF_OF_GUINEA: "Gulf of Guinea",
        #         Location.GULF_OF_THE_FARALLONES: "Gulf of the Farallones",
        #         Location.NARINDA_BAY: "Narinda Bay",
        #         Location.WESTERN_PACIFIC_OCEAN_NORTH: "Western North Pacific Ocean",
        #         Location.ONTONG_JAVA_PLATEAU: "Ontong Java Plateau",
        #         Location.AMUNDSEN_SEA: "Amundsen Sea",
        #         Location.SANTOS_BASIN: "Santos Basin",
        #         Location.EASTERN_PACIFIC_OCEAN_SOUTH: "Eastern South Pacific Ocean",
        #         Location.STRAIT_OF_GEORGIA: "Strait of Georgia",
        #         Location.STRAIT_OF_HORMUZ: "Strait of Hormuz",
        #         Location.TANPA_BAY: "Tampa Bay",
        #         Location.RIA_DE_VIGO: "Ria of Vigo",
        #     }
        #     return names[self]

        # else:
        #     raise NotImplementedError(
        #         f"Localization not available for language '{language}'."
        #     )

    def is_shallow_water(self) -> bool:
        """ Indicates whether the location is classified as shallow water. """
        shallow_sites = {
            Location.ADRIATIC_SEA,
            Location.NORTH_SEA,
            Location.GUANABARA_BAY,
            Location.GULF_OF_THE_FARALLONES,
            Location.NARINDA_BAY,
            Location.STRAIT_OF_HORMUZ,
            Location.TANPA_BAY,
            Location.RIA_DE_VIGO,
        }
        return self in shallow_sites

    def seabed_type(self) -> lps_layer.SeabedType:
        """ Return the Seabed Type for the Location """
        seabed_map = {
            Location.ADRIATIC_SEA: lps_layer.SeabedType.LIMESTONE, #https://doi.org/10.1016/j.marpetgeo.2015.03.015
            Location.AMUNDSEN_SEA: lps_layer.SeabedType.MORAINE, #https://doi.org/10.1144/M46.183
            Location.ANDFJORDEN: lps_layer.SeabedType.MORAINE, #https://doi.org/10.1016/j.margeo.2015.02.001
            Location.BENGAL_BAY: lps_layer.SeabedType.SILT,
            Location.EASTERN_PACIFIC_OCEAN_SOUTH: lps_layer.SeabedType.CLAY,
            Location.GUANABARA_BAY: lps_layer.SeabedType.CLAY,
            Location.GULF_OF_GUINEA: lps_layer.SeabedType.SILT,
            Location.GULF_OF_PENAS: lps_layer.SeabedType.GRAVEL,
            Location.GULF_OF_ST_LAWRENCE: lps_layer.SeabedType.SAND,
            Location.GULF_OF_THE_FARALLONES: lps_layer.SeabedType.SAND,
            Location.MOZAMBIQUE_CHANNEL: lps_layer.SeabedType.SAND,
            Location.NORTH_SEA: lps_layer.SeabedType.CHALK, #https://doi.org/10.1029/2011JB008564
            Location.ONTONG_JAVA_PLATEAU: lps_layer.SeabedType.CHALK, #https://doi.org/10.1029/JB083iB01p00283
            Location.RIA_DE_VIGO: lps_layer.SeabedType.CLAY,
            Location.SANTOS_BASIN: lps_layer.SeabedType.CLAY,
            Location.STRAIT_OF_GEORGIA: lps_layer.SeabedType.SAND,
            Location.STRAIT_OF_HORMUZ: lps_layer.SeabedType.SAND,
            Location.TANPA_BAY: lps_layer.SeabedType.LIMESTONE, #https://doi.org/10.1016/S0025-3227(03)00189-0
            Location.WESTERN_PACIFIC_OCEAN_NORTH: lps_layer.SeabedType.CLAY,
            Location.YUCATAN_BASIN: lps_layer.SeabedType.BASALT,
        }
        return seabed_map[self]


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
        for local in Location:
            data.append(local.as_dict())

        df = pd.DataFrame(data)
        return df

    @staticmethod
    def rand() -> "Location":
        """Return a random Local."""
        return random.choice(list(Location))

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

    def __init__(self, local: Location = None, month: Month = None):
        self.local = local or Location.rand()
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
