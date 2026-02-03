"""
Collection Module

Define Collections and their generation process.
"""
import os
import typing
import random
import copy
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as sci_wave
import scipy.signal as sci_sig

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.scenario as lps_scenario
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.environment.acoustic_site as lps_as

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
                 dynamic_id: int):
        self.ship_id = ship_id
        self.scenario_id = scenario_id
        self.dynamic_id = dynamic_id

    def __str__(self):
        return f"[{self.ship_id}] {self.scenario_id} | {self.dynamic_id}"

    def as_dict(self) -> dict[str, typing.Any]:
        """ Converts the entry into a dictionary suitable for tabular representation. """
        return {
                "SHIP_CATALOG_ID": self.ship_id,
                "SCENARIO_CATALOG_ID": self.scenario_id,
                "DYNAMIC_CATALOG_ID": self.dynamic_id,
            }

    @staticmethod
    def load_entries(filename: str) -> typing.List["DatabaseEntry"]:
        df = pd.read_csv(filename)
        entries = []

        for _, row in df.iterrows():
            entries.append(
                DatabaseEntry(
                    ship_id=int(row["SHIP_CATALOG_ID"]),
                    scenario_id=int(row["SCENARIO_CATALOG_ID"]),
                    dynamic_id=int(row["DYNAMIC_CATALOG_ID"]),
                )
            )

        return entries

class Database(syndb_core.Catalog[DatabaseEntry]):
    """Defines a dataset catalog that links ships, scenarios, and dynamics."""

    def __init__(self,
                 ship_catalog: syndb_ship.ShipCatalog,
                 acoutic_scenario_catalog: syndb_core.Catalog[syndb_scenario.AcousticScenario],
                 n_samples: int,
                 seed: int = 42):

        rng = random.Random(seed)
        entries = []

        n_ships = len(ship_catalog)
        n_scenarios = len(acoutic_scenario_catalog)

        dynamic_catalog = syndb_dynamic.SimulationDynamic.rand_catalog(n_samples=n_samples)

        for i in range(n_samples):

            ship_ids = rng.sample(range(n_ships), 2)
            scenario_ids = rng.sample(range(n_scenarios), 2)

            for ship_id in ship_ids:
                for scenario_id in scenario_ids:
                    entries.append(
                        DatabaseEntry(
                            ship_id=ship_id,
                            scenario_id=scenario_id,
                            dynamic_id=i
                        )
                    )

        super().__init__(entries=entries)
        self.ship_catalog = ship_catalog
        self.acoutic_scenario_catalog = acoutic_scenario_catalog
        self.dynamic_catalog = dynamic_catalog

    def export(self, output_dir: str) -> None:
        """ Exports the catalog and associated tables to CSV files. """
        os.makedirs(output_dir, exist_ok=True)

        df = self.to_df()
        df.to_csv(os.path.join(output_dir, "database.csv"), index=False, encoding="utf-8")

        df = self.ship_catalog.to_df()
        df.to_csv(os.path.join(output_dir, "ship_catalog.csv"), index=False, encoding="utf-8")

        df = self.acoutic_scenario_catalog.to_df()
        df.to_csv(os.path.join(output_dir, "acoutic_scenario_catalog.csv"),
                  index=False,
                  encoding="utf-8")

        df = self.dynamic_catalog.to_df()
        df.to_csv(os.path.join(output_dir, "dynamic_catalog.csv"), index=False, encoding="utf-8")


    @classmethod
    def load(cls, output_dir: str) -> "Database":

        ship_catalog = syndb_ship.ShipCatalog.load(
            os.path.join(output_dir, "ship_catalog.csv")
        )

        scenario_catalog = syndb_scenario.AcousticScenario.load_catalog(
            os.path.join(output_dir, "acoutic_scenario_catalog.csv")
        )

        dynamic_catalog = syndb_dynamic.SimulationDynamic.load_catalog(
            os.path.join(output_dir, "dynamic_catalog.csv")
        )

        entries = DatabaseEntry.load_entries(os.path.join(output_dir, "database.csv"))

        db = cls.__new__(cls)
        super(Database, db).__init__(entries=entries)

        db.ship_catalog = ship_catalog
        db.acoutic_scenario_catalog = scenario_catalog
        db.dynamic_catalog = dynamic_catalog

        return db

    def synthesize(self,
                   output_dir: str,
                   sonar: lps_sonar.Sonar,
                   sample_frequency: lps_qty.Frequency,
                   step_interval: lps_qty.Time =lps_qty.Time.s(1),
                   simulation_steps: int = 10,
                   global_attenuation_dB: float = 20,
                   only_plot: bool = False,
                   force_override: bool = False) -> None:

        for i in tqdm.tqdm(range(len(self)), desc="Synthesizing", leave=False, ncols=120):
            self.synthesize_sample(
                sample_index=i,
                output_dir=output_dir,
                sonar=sonar,
                sample_frequency=sample_frequency,
                step_interval=step_interval,
                simulation_steps=simulation_steps,
                global_attenuation_dB=global_attenuation_dB,
                only_plot = only_plot,
                force_override = force_override
            )

    def synthesize_sample(self,
                   sample_index: int,
                   output_dir: str,
                   sonar: lps_sonar.Sonar,
                   sample_frequency: lps_qty.Frequency,
                   step_interval: lps_qty.Time,
                   simulation_steps: int,
                   global_attenuation_dB: float,
                   only_plot: bool,
                   force_override: bool) -> None:

        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f"{sample_index}.wav")
        dynamic_geo_spec = os.path.join(output_dir, f"{sample_index}_spec.png")
        dynamic_geo_file = os.path.join(output_dir, f"{sample_index}_geo.png")
        dynamic_dist_file = os.path.join(output_dir, f"{sample_index}_dist.png")

        if os.path.exists(filename) and not force_override:
            return

        entry = self[sample_index]
        acoustic_scenario = self.acoutic_scenario_catalog[entry.scenario_id]
        ship_info = self.ship_catalog[entry.ship_id]
        dynamic = self.dynamic_catalog[entry.dynamic_id]

        channel = acoustic_scenario.get_channel()
        environment = acoustic_scenario.get_env()
        environment.global_attenuation_dB = global_attenuation_dB

        scenario = lps_scenario.Scenario(
            step_interval=step_interval,
            channel = channel,
            environment = environment
        )

        sonar_i = copy.deepcopy(sonar)

        scenario.add_sonar("main", sonar_i)

        ship = ship_info.make_ship(
            dynamic=dynamic,
            step_interval=step_interval,
            simulation_steps=simulation_steps
        )

        scenario.add_noise_container(ship)

        sonar_i.ref_state = dynamic.get_sonar_initial_state(ship)

        scenario.simulate(simulation_steps)

        scenario.geographic_plot(dynamic_geo_file)
        scenario.relative_distance_plot(dynamic_dist_file)

        if not only_plot:

            signal = scenario.get_sonar_audio("main", fs=sample_frequency)

            sci_wave.write(filename, int(sample_frequency.get_hz()), signal)


            float_signal = signal.astype(np.float32) / (2**15 -1)
            f, t, s = sci_sig.spectrogram(float_signal[:,0],
                                            fs=sample_frequency.get_hz(),
                                            nperseg=2048)

            plt.figure(figsize=(10, 6))
            plt.pcolormesh(t, f, 20 * np.log10(np.clip(s, 1e-10, None)), shading='gouraud')
            plt.ylabel('Frequência [Hz]')
            plt.xlabel('Tempo [s]')
            plt.colorbar(label='Intensidade [dB]')
            plt.savefig(dynamic_geo_spec)
            plt.close()



class ToyDatabase(Database):
    """Small-scale catalog for demonstration or testing purposes."""

    def __init__(self, n_samples = 100, seed: int = 42):
        selected_locals = [
            syndb_scenario.Location.GUANABARA_BAY
        ]

        rng = random.Random(seed)
        scenarios = []
        for local in selected_locals:
            seasons = rng.sample(list(lps_as.Season), 2)
            for season in seasons:
                scenarios.append(syndb_scenario.AcousticScenario(local, season))

        super().__init__(
            ship_catalog=syndb_ship.ShipCatalog(),
            acoutic_scenario_catalog=syndb_core.Catalog[syndb_scenario.AcousticScenario](entries=scenarios),
            n_samples=n_samples,
            seed=seed
        )

class OlocumDatabase(Database):
    """Comprehensive simulated catalog inspired by the OLOCUM dataset."""

    def __init__(self, n_scenarios = 100, n_ships = 50, n_samples = 1000, seed: int = 42):

        rng = random.Random(seed)
        all_scenarios = [
            syndb_scenario.AcousticScenario(local, season)
            for local in syndb_scenario.Location
            for season in lps_as.Season
        ]
        n_scenarios = min(n_scenarios, len(all_scenarios))
        scenarios = rng.sample(all_scenarios, n_scenarios)

        super().__init__(
            ship_catalog=syndb_ship.ShipCatalog(n_samples=n_ships, seed=seed),
            acoutic_scenario_catalog=syndb_core.Catalog[syndb_scenario.AcousticScenario](entries=scenarios),
            n_samples=n_samples,
            seed=seed
        )
