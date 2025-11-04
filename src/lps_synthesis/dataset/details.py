import enum
import os
import typing
import random

import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_sce_dyn
import lps_synthesis.dataset.dynamic as lps_db_dyn

class Local(enum.Enum):
    """ Enumeration of reference oceanic and coastal locations. """

    GUANABARA_BAY = enum.auto()
    VIGO_PORT = enum.auto()
    QIANDAO_LAKE = enum.auto()
    DOGGER_BANK = enum.auto()
    CAMPECHE_BAY = enum.auto()
    TOKYO_BAY = enum.auto()
    STRAIT_OF_HORMUZ = enum.auto()
    EXMOUTH_GULF = enum.auto()
    GULF_OF_GUINEA = enum.auto()
    GULF_OF_THE_FARALLONES = enum.auto()
    SANTOS_BASIN = enum.auto()
    STRAIT_OF_GEORGIA = enum.auto()
    BERMUDA_RISE = enum.auto()
    WALVIS_RIDGE = enum.auto()
    MARIANA_BASIN = enum.auto()
    HAWAII_RIDGE = enum.auto()
    DRAKE_PASSAGE = enum.auto()
    ARABIAN_SEA = enum.auto()
    TASMAN_SEA = enum.auto()
    BARENTS_SEA = enum.auto()

    def get_point(self) -> lps_sce_dyn.Point:
        """ Returns the latitude and longitude of the selected location. """
        latlon_dict = {
            Local.GUANABARA_BAY: [-22.93, -43.14],
            Local.VIGO_PORT: [42.25, -8.73],
            Local.QIANDAO_LAKE: [29.56, 118.97],
            Local.DOGGER_BANK: [55.30, 3.00],
            Local.CAMPECHE_BAY: [19.30, -92.00],
            Local.TOKYO_BAY: [35.40, 139.80],
            Local.STRAIT_OF_HORMUZ: [26.50, 56.50],
            Local.EXMOUTH_GULF: [-21.80, 114.10],
            Local.GULF_OF_GUINEA: [4.38, 7.07],
            Local.GULF_OF_THE_FARALLONES: [37.76, -122.85],
            Local.SANTOS_BASIN: [-25.00, -43.00],
            Local.STRAIT_OF_GEORGIA: [49.08, -123.34],
            Local.BERMUDA_RISE: [31.80, -64.70],
            Local.WALVIS_RIDGE: [-25.50, 3.00],
            Local.MARIANA_BASIN: [14.60, 146.30],
            Local.HAWAII_RIDGE: [20.00, -157.00],
            Local.DRAKE_PASSAGE: [-60.50, -56.00],
            Local.ARABIAN_SEA: [15.00, 66.00],
            Local.TASMAN_SEA: [-40.00, 165.00],
            Local.BARENTS_SEA: [72.00, 40.00],
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
            Local.GUANABARA_BAY,
            Local.VIGO_PORT,
            Local.QIANDAO_LAKE,
            Local.DOGGER_BANK,
            Local.CAMPECHE_BAY,
            Local.TOKYO_BAY,
            Local.STRAIT_OF_HORMUZ,
            Local.EXMOUTH_GULF,
            Local.GULF_OF_GUINEA,
            Local.GULF_OF_THE_FARALLONES,
        }
        return self in shallow_sites

    def __str__(self):
        return self.to_string()

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
            p = local.get_point()
            data.append({
                "ID": local.value,
                "Name (us)": local.to_string(),
                "Latitude (deg)": p.latitude.get_deg(),
                "Longitude (deg)": p.longitude.get_deg(),
                "Latitude (dms)": str(p.latitude),
                "Longitude (dms)": str(p.longitude),
            })

        df = pd.DataFrame(data)
        return df

    @staticmethod
    def rand() -> "Local":
        """Return a random Local."""
        return random.choice(list(Local))

class Month(enum.IntEnum):
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

class AcousticScenario:

    def __init__(self, local: Local = None, month: Month = None):
        self.local = local or Local.rand()
        self.month = month or Month.rand()

    def __str__(self):
        return f"{self.local} [{self.month.name.capitalize()}]"

class CatalogEntry:

    def __init__(self,
                 scenario: AcousticScenario = None,
                 dynamic: lps_db_dyn.SimulationDynamic = None):
        self.scenario = scenario or AcousticScenario()
        self.dynamic = dynamic or lps_db_dyn.SimulationDynamic.rand()

    def __str__(self):
        return f"{self.scenario} | {self.dynamic}"

class Catalog:

    def __init__(self, scenarios: typing.List[CatalogEntry] = None):
        self._entries = scenarios or []

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        for el in self._entries:
            yield el

    def __getitem__(self, index):
        return self._entries[index]

    def to_df(self) -> pd.DataFrame:
        """Convert all scenarios into a DataFrame."""
        data = []
        for entry in self._entries:
            data.append({
                "Local ID": entry.scenario.local.value,
                "Local Name": entry.scenario.local.to_string(),
                "Month": entry.scenario.month.name.capitalize(),
                "Dynamic": str(entry.dynamic.dynamic_type),
                "Shortest Dist (m)": entry.dynamic.shortest.get_m()
            })
        df = pd.DataFrame(data)
        return df


class ToyCatalog(Catalog):

    def __init__(self, n_samples = 100, seed: int = 42):
        selected_locals = [
            Local.GUANABARA_BAY,
            Local.SANTOS_BASIN,
            Local.VIGO_PORT,
            Local.STRAIT_OF_GEORGIA,
            Local.QIANDAO_LAKE,
        ]

        random.seed(seed)
        scenarios = []
        for local in selected_locals:
            months = random.sample(list(Month), 2)
            for month in months:
                scenarios.append(AcousticScenario(local, month))

        random.seed(seed)
        super().__init__([CatalogEntry(random.choice(scenarios)) for _ in range(n_samples)])


class OlocumCatalog(Catalog):

    def __init__(self, n_scenarios = 100, n_samples = 1000, seed: int = 42):
        random.seed(seed)
        all_scenarios = [
            AcousticScenario(local, month)
            for local in Local
            for month in Month
        ]
        n_scenarios = min(n_scenarios, len(all_scenarios))
        scenarios = random.sample(all_scenarios, n_scenarios)

        random.seed(seed)
        super().__init__([CatalogEntry(random.choice(scenarios)) for _ in range(n_samples)])


def _main():
    out_dir = "./result"
    os.makedirs(out_dir, exist_ok=True)

    Local.plot(os.path.join(out_dir, "locals.png"))
    local_df = Local.to_df()
    local_df.to_csv(os.path.join(out_dir, "locals.csv"), index=False, encoding="utf-8")

    toy_dataset = ToyCatalog()
    toy_dataset_df = toy_dataset.to_df()
    toy_dataset_df.to_csv(os.path.join(out_dir, "toy_dataset.csv"), index=False, encoding="utf-8")

    olocum_dataset = OlocumCatalog()
    olocum_dataset_df = olocum_dataset.to_df()
    olocum_dataset_df.to_csv(os.path.join(out_dir, "olocum_dataset.csv"),
                             index=False,
                             encoding="utf-8")

    print()
    print("############")
    print(local_df)

    count = toy_dataset_df.groupby(['Local ID','Month']).size().reset_index(name="Qty")
    print()
    print("############")
    print(count)

    count = toy_dataset_df.groupby(['Dynamic']).size().reset_index(name="Qty")
    print()
    print("############")
    print(count)

    count = olocum_dataset_df.groupby(['Local ID','Month']).size().reset_index(name="Qty")
    count = count.sort_values("Qty", ascending=False)
    print()
    print("############")
    print(count)

    count = olocum_dataset_df.groupby(['Dynamic']).size().reset_index(name="Qty")
    print()
    print("############")
    print(count)

if __name__ == "__main__":
    _main()
