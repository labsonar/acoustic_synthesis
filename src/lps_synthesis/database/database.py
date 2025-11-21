"""
Collection Module

Define Collections and their generation process.
"""
import os
import typing
import random
import tqdm

import scipy.io.wavfile as sci_wave

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.scenario as lps_scenario
import lps_synthesis.scenario.sonar as lps_sonar

import lps_synthesis.database.dynamic as syndb_dynamic
import lps_synthesis.database.scenario as syndb_scenario
import lps_synthesis.database.ship as syndb_ship

import lps_synthesis.database.catalog as syndb_core

class DatabaseEntry(syndb_core.CatalogEntry):
    """Represents a single entry in a catalog linking
        a ship, a scenario, and a dynamic configuration."""

    def __init__(self,
                 ship_id: int,
                 scenario_id: int,
                 dynamic: syndb_dynamic.SimulationDynamic = None):
        self.ship_id = ship_id
        self.scenario_id = scenario_id
        self.dynamic = dynamic or syndb_dynamic.SimulationDynamic.rand()

    def __str__(self):
        return f"[{self.ship_id}] {self.scenario_id} | {self.dynamic}"

    def as_dict(self) -> dict[str, typing.Any]:
        """ Converts the entry into a dictionary suitable for tabular representation. """
        return {
                "Ship ID": self.ship_id,
                "Scenario ID": self.scenario_id,
                "Dynamic": str(self.dynamic.dynamic_type),
                "Shortest Dist (m)": self.dynamic.shortest.get_m()
            }


class Database(syndb_core.Catalog[DatabaseEntry]):
    """Defines a dataset catalog that links ships, scenarios, and dynamics."""

    def __init__(self,
                 ship_catalog: syndb_ship.ShipCatalog,
                 scenario_catalog: syndb_core.Catalog[syndb_scenario.AcousticScenario],
                 n_samples: int,
                 seed: int = 42):

        rng = random.Random(seed)
        entries = [DatabaseEntry(ship_id=rng.randrange(len(ship_catalog)),
                                   scenario_id=rng.randrange(len(scenario_catalog)))
                    for _ in range(n_samples)]

        super().__init__(entries=entries)
        self.ship_catalog = ship_catalog
        self.scenario_catalog = scenario_catalog

    def export(self, output_dir: str) -> None:
        """ Exports the catalog and associated tables to CSV files. """
        os.makedirs(output_dir, exist_ok=True)

        df = self.to_df()
        df.to_csv(os.path.join(output_dir, "database.csv"), index=False, encoding="utf-8")

        df = self.ship_catalog.to_df()
        df.to_csv(os.path.join(output_dir, "ship_catalog.csv"), index=False, encoding="utf-8")

        df = self.scenario_catalog.to_df()
        df.to_csv(os.path.join(output_dir, "scenario_catalog.csv"), index=False, encoding="utf-8")

    def synthesize(self,
                   output_dir: str,
                   sonar: lps_sonar.Sonar,
                   sample_frequency: lps_qty.Frequency,
                   step_interval: lps_qty.Time =lps_qty.Time.s(1),
                   simulation_steps: int = 10) -> None:

        os.makedirs(output_dir, exist_ok=True)

        for i, entry in enumerate(tqdm.tqdm(self, desc="Synthesizing", leave=False, ncols=120)):
            scenario = self.scenario_catalog[entry.scenario_id]
            ship_info = self.ship_catalog[entry.ship_id]

            channel = scenario.get_channel()
            environment = scenario.get_env()

            scenario = lps_scenario.Scenario(
                step_interval=step_interval,
                channel = channel,
                environment = environment
            )

            scenario.add_sonar("main", sonar)

            ship = ship_info.make_ship(dynamic=entry.dynamic,
                                       interval=simulation_steps * step_interval)

            scenario.add_noise_container(ship)

            sonar.ref_state = entry.dynamic.get_sonar_initial_state(ship)

            scenario.simulate(simulation_steps)

            signal = scenario.get_sonar_audio("main", fs=sample_frequency)

            filename = os.path.join(output_dir, f"{i}.wav")
            sci_wave.write(filename, int(sample_frequency.get_hz()), signal)

class ToyDatabase(Database):
    """Small-scale catalog for demonstration or testing purposes."""

    def __init__(self, n_samples = 100, seed: int = 42):
        selected_locals = [
            syndb_scenario.Location.GUANABARA_BAY,
            syndb_scenario.Location.SANTOS_BASIN,
            syndb_scenario.Location.VIGO_PORT,
            syndb_scenario.Location.STRAIT_OF_GEORGIA,
            syndb_scenario.Location.QIANDAO_LAKE,
        ]

        rng = random.Random(seed)
        scenarios = []
        for local in selected_locals:
            months = rng.sample(list(syndb_scenario.Month), 2)
            for month in months:
                scenarios.append(syndb_scenario.AcousticScenario(local, month))

        super().__init__(
            ship_catalog=syndb_ship.ShipCatalog(),
            scenario_catalog=syndb_core.Catalog[syndb_scenario.AcousticScenario](entries=scenarios),
            n_samples=n_samples,
            seed=seed
        )

class OlocumDatabase(Database):
    """Comprehensive simulated catalog inspired by the OLOCUM dataset."""

    def __init__(self, n_scenarios = 100, n_ships = 50, n_samples = 1000, seed: int = 42):

        rng = random.Random(seed)
        all_scenarios = [
            syndb_scenario.AcousticScenario(local, month)
            for local in syndb_scenario.Location
            for month in syndb_scenario.Month
        ]
        n_scenarios = min(n_scenarios, len(all_scenarios))
        scenarios = rng.sample(all_scenarios, n_scenarios)

        super().__init__(
            ship_catalog=syndb_ship.ShipCatalog(n_samples=n_ships, seed=seed),
            scenario_catalog=syndb_core.Catalog[syndb_scenario.AcousticScenario](entries=scenarios),
            n_samples=n_samples,
            seed=seed
        )
