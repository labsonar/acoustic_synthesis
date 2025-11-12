"""
Ship Module

Define ShipInfo and ShipCatalog.
"""
import os
import typing
import random

import pandas as pd

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.noise_source as lps_ns
import lps_synthesis.database.dynamic as syndb_dynamic

import lps_synthesis.database.catalog as syndb_core

class ShipInfo(syndb_core.CatalogEntry):
    """Container class for ship information."""

    def __init__(
        self,
        ship_id: int,
        ship_type: lps_ns.ShipType,
        iara_ship_id: str,
        ship_name: str,
        mcr_percent: float,
        cruising_speed: lps_qty.Speed,
        rotacional_frequency: lps_qty.Frequency,
        length: lps_qty.Distance,
        draft: lps_qty.Distance,
        n_blades: int,
        n_shafts: int,
    ):
        self.ship_id = ship_id
        self.ship_type = ship_type
        self.iara_ship_id = iara_ship_id
        self.ship_name = ship_name
        self.mcr_percent = mcr_percent
        self.cruising_speed = cruising_speed
        self.max_speed = self.cruising_speed/self.mcr_percent
        self.rotacional_frequency = rotacional_frequency
        self.length = length
        self.draft = draft
        self.n_blades = n_blades
        self.n_shafts = n_shafts
        self.rng = random.Random(self.ship_id)
        self.sigma_factor = self.rng.randint(10, 1000)/1000

    def __repr__(self):
        return f"Ship {self.iara_ship_id} <{self.ship_name}>"

    def random_speed(self) -> lps_qty.Speed:
        """ Generate a randomized speed. """
        sigma = self.sigma_factor * self.cruising_speed.get_kt()
        value = self.rng.gauss(self.cruising_speed.get_kt(), sigma)
        return lps_qty.Speed.kt(max(0, min(value, self.max_speed.get_kt())))

    def make_ship(self,
                  dynamic: syndb_dynamic.SimulationDynamic,
                  interval: lps_qty.Time) -> lps_ns.Ship:
        """ Allocate the lps_ns.Ship based on ShipInfo. """

        return lps_ns.Ship(
            ship_id=self.ship_id,
            propulsion=lps_ns.CavitationNoise(
                ship_type=self.ship_type,
                n_blades=self.n_blades,
                n_shafts=self.n_shafts,
                length=self.length,
                cruise_speed=self.cruising_speed,
                cruise_rotacional_frequency=self.rotacional_frequency,
                max_speed=self.max_speed,
                seed=self.ship_id
            ),
            draft=self.draft,
            initial_state=dynamic.get_ship_initial_state(speed=self.random_speed(),
                                                    interval=interval),
            seed=self.ship_id
        )

    def as_dict(self):
        return {
            "ship_id": self.ship_id,
            "ship_type": self.ship_type.name,
            "iara_ship_id": self.iara_ship_id,
            "ship_name": self.ship_name,
            "mcr_percent": self.mcr_percent,
            "max_speed_kt": self.max_speed.get_kt(),
            "cruising_speed_kt": self.cruising_speed.get_kt(),
            "rpm": self.rotacional_frequency.get_rpm(),
            "length_m": self.length.get_m(),
            "draft_m": self.draft.get_m(),
            "n_blades": self.n_blades,
            "n_shafts": self.n_shafts,
            "sigma_factor": self.sigma_factor,
        }

class ShipCatalog(syndb_core.Catalog[ShipInfo]):
    """Catalog of ships with optional random sampling and range variation."""

    NUMERIC_FIELDS = [
        "mcr_percent",
        "max_speed_kt",
        "cruising_speed_kt",
        "rpm",
        "length_m",
        "draft_m",
        "n_blades",
        "n_shafts",
    ]

    def __init__(self, n_samples: typing.Optional[int] = None, seed: int = 42):

        csv_path = os.path.join(os.path.dirname(__file__), "data", "ship_info.csv")
        df = pd.read_csv(csv_path)

        if n_samples is None:
            rows = df
        else:
            rows = df.sample(n=n_samples, replace=True, random_state=seed)

        ships = [self._make_shipinfo(row, i, seed) for i, (_, row) in enumerate(rows.iterrows())]

        super().__init__(entries=ships)
        self.df = df

    @staticmethod
    def _parse_value(value: typing.Union[str, float, int], seed: int) -> typing.Union[float, int]:
        """ Parse a single value or range like '10-15', returning a random value. """
        if isinstance(value, (float, int)):
            return value
        if not isinstance(value, str) or value.strip() == "":
            return None

        value = value.strip()
        if "-" in value:
            parts = value.split("-")
            a = float(parts[0]) if "." in parts[0] else int(parts[0])
            b = float(parts[1]) if "." in parts[1] else int(parts[1])

            rng = random.Random(seed)

            if isinstance(a, int) and isinstance(b, int):
                return rng.randint(a, b)
            return int(rng.uniform(float(a), float(b)) * 100)/100
        else:
            try:
                if "." in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return None

    @staticmethod
    def parse_ship_type(name: str) -> lps_ns.ShipType:
        """Convert a string into a ShipType enum member."""
        if not isinstance(name, str):
            return lps_ns.ShipType.OTHER

        normalized = name.strip().upper().replace(" ", "_")
        try:
            return lps_ns.ShipType[normalized]
        except KeyError:
            return lps_ns.ShipType.OTHER

    def _make_shipinfo(self, row: pd.Series, ship_id: int, seed: int) -> ShipInfo:
        """Build a ShipInfo object from a dataframe row."""
        data = {}

        for col in row.index:
            val = row[col]
            if col in self.NUMERIC_FIELDS:
                data[col] = self._parse_value(val, seed + ship_id)
            else:
                data[col] = val

        return ShipInfo(
            ship_id = ship_id,
            ship_type = ShipCatalog.parse_ship_type(data["ship_type"]),
            iara_ship_id = data["iara_ship_id"],
            ship_name = data["ship_name"],
            mcr_percent = data["mcr_percent"],
            cruising_speed = lps_qty.Speed.kt(data["cruising_speed_kt"]),
            rotacional_frequency = lps_qty.Frequency.rpm(data["rpm"]),
            length = lps_qty.Distance.m(data["length_m"]),
            draft = lps_qty.Distance.m(data["draft_m"]),
            n_blades = data["n_blades"],
            n_shafts = data["n_shafts"],
        )
