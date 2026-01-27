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
        max_speed: lps_qty.Speed,
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
        self.mcr_percent = int(cruising_speed/max_speed * 100) / 100
        self.cruising_speed = cruising_speed
        self.max_speed = max_speed
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
            ship_id=f"{self.ship_id}",
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
            "SHIP_ID": self.ship_id,
            "SHIP_TYPE": self.ship_type.name,
            "IARA_SHIP_ID": self.iara_ship_id,
            "SHIP_NAME": self.ship_name,
            "MCR_PERCENT": self.mcr_percent,
            "MAX_SPEED_KT": self.max_speed.get_kt(),
            "CRUISING_SPEED_KT": self.cruising_speed.get_kt(),
            "RPM": self.rotacional_frequency.get_rpm(),
            "LENGTH_M": self.length.get_m(),
            "DRAFT_M": self.draft.get_m(),
            "N_BLADES": self.n_blades,
            "N_SHAFTS": self.n_shafts,
            "SIGMA_FACTOR": self.sigma_factor,
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
    def _parse_value(value: typing.Union[str, float, int], seed: int) -> \
            typing.Union[float, int, None]:
        """
        Parse a value that can be:
        - a single number: "10" or 10
        - a range: "10-15"
        - a list of values/ranges: "10, 12-18, 25, 3.7"
        Returns one random value selected from the expanded set.
        """

        if isinstance(value, (int, float)):
            return value

        if not isinstance(value, str):
            return None

        value = value.strip()
        if not value:
            return None

        rng = random.Random(seed)

        if "/" in value:
            options = [v.strip() for v in value.split("/")]
            parsed = []

            for opt in options:
                if "-" in opt:
                    parts = opt.split("-")
                    if len(parts) != 2:
                        continue
                    try:
                        a = float(parts[0])
                        b = float(parts[1])
                    except ValueError:
                        continue

                    if a.is_integer() and b.is_integer():
                        parsed.extend(range(int(a), int(b) + 1))
                    else:
                        parsed.append(round(rng.uniform(a, b), 2))
                else:
                    try:
                        parsed.append(int(opt) if opt.isdigit() else float(opt))
                    except ValueError:
                        continue

            return rng.choice(parsed) if parsed else None

        if "-" in value:
            parts = value.split("-")
            if len(parts) != 2:
                return None
            try:
                a = float(parts[0])
                b = float(parts[1])
            except ValueError:
                return None

            if a.is_integer() and b.is_integer():
                return rng.randint(int(a), int(b))
            return round(rng.uniform(a, b), 2)

        try:
            return int(value) if value.isdigit() else float(value)
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
            max_speed = lps_qty.Speed.kt(data["max_speed_kt"]),
            cruising_speed = lps_qty.Speed.kt(data["cruising_speed_kt"]),
            rotacional_frequency = lps_qty.Frequency.rpm(data["rpm"]),
            length = lps_qty.Distance.m(data["length_m"]),
            draft = lps_qty.Distance.m(data["draft_m"]),
            n_blades = data["n_blades"],
            n_shafts = data["n_shafts"],
        )
