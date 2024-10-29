"""
Module for representing the scenario and their elements
"""
import enum
import typing
import math
import random
import overrides
import tqdm

import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.channel as lps_channel
import lps_synthesis.propagation.channel_description as lps_desc
import lps_sp.acoustical.broadband as lps_bb

class ShipType(enum.Enum):
    " Enum class representing the possible ship types (https://www.mdpi.com/2077-1312/9/4/369)"
    BULKER = 0
    CONTAINERSHIP = 1
    CRUISE = 2
    DREDGER = 3
    FISHING = 4
    GOVERNMENT = 5
    RESEARCH = 5
    NAVAL = 6
    PASSENGER = 7
    RECREATIONAL = 8
    TANKER = 9
    TUG = 10
    VEHICLE_CARRIER = 11
    OTHER = 12

    def __str__(self):
        return self.name.lower()

    def _get_ref_speed(self) -> lps_qty.Speed:
        speed_ref = {
            ShipType.BULKER: 13.9,
            ShipType.CONTAINERSHIP: 18.0,
            ShipType.CRUISE: 17.1,
            ShipType.DREDGER: 9.5,
            ShipType.FISHING: 6.4,
            ShipType.GOVERNMENT: 8.0,
            ShipType.NAVAL: 11.1,
            ShipType.PASSENGER: 9.7,
            ShipType.RECREATIONAL: 10.6,
            ShipType.TANKER: 12.4,
            ShipType.TUG: 3.7,
            ShipType.VEHICLE_CARRIER: 15.8,
            ShipType.OTHER: 7.4,
        }
        return lps_qty.Speed.kt(speed_ref[self])

    def _is_cargo(self) -> bool:
        cargo_ships = [ShipType.CONTAINERSHIP,
                       ShipType.VEHICLE_CARRIER,
                       ShipType.BULKER,
                       ShipType.TANKER]
        return self in cargo_ships

    def to_psd(self,
                fs: lps_qty.Frequency,
                lower_bound: lps_qty.Frequency = lps_qty.Frequency.hz(2),
                lenght: lps_qty.Distance = lps_qty.Distance.ft(300),
                speed: lps_qty.Speed = None) -> \
                    typing.Tuple[typing.List[lps_qty.Frequency], np.array]:
        """Return frequencies and corresponding PDS.
            An implementation of JOMOPANS-ECHO Model (https://www.mdpi.com/2077-1312/9/4/369)

        Args:
            fs (lps_qty.Frequency): Sample frequency
            lower_bound (lps_qty.Frequency, optional): Lower Frequency to be computed.
                Defaults to 2 Hz.
            lenght (lps_qty.Distance, optional): Ship lenght. Defaults to 300 ft.
            speed (lps_qty.Speed, optional): Ship speed. Defaults based on class.

        Returns:
            typing.Tuple[
                typing.List[lps_qty.Frequency],  : frequencies
                np.array]                        : PDS in (dB re 1 µPa^2/Hz @1m)
            ]
        """

        if speed is None:
            speed = self._get_ref_speed()

        frequencies = []
        f_ref = lps_qty.Frequency.khz(1)

        f_index = math.ceil(3 * math.log(lower_bound / f_ref, 2))
        # lower index in 1/3 octave inside interval

        while True:
            fi = f_ref * 2**(f_index / 3)

            if fi > fs/2:
                break

            frequencies.append(fi)
            f_index += 1

        l_0 = lps_qty.Distance.ft(300)
        v_c = self._get_ref_speed()
        v_ref = lps_qty.Speed.kt(1)
        f_ref = lps_qty.Frequency.hz(1)

        psd = np.zeros(len(frequencies)) + 60*math.log10(speed/v_c) + 20*math.log10(lenght/l_0)

        for index, f in enumerate(frequencies):
            if f < lps_qty.Frequency.hz(100) and self._is_cargo():
                k_lf = 208
                if self == ShipType.CONTAINERSHIP or self == ShipType.BULKER:
                    d_lf = 0.8
                else:
                    d_lf = 1
                f_1_lf = lps_qty.Frequency.hz(600) * (v_ref/v_c)

                psd[index] += k_lf - 40*math.log10(f_1_lf/f_ref) + 10*math.log10(f/f_ref) \
                              - 10*math.log10((1 - (f/f_1_lf)**2)**2 + d_lf**2)

            else:
                f_1 = lps_qty.Frequency.hz(480) * (v_ref/v_c)
                k = 191
                d = 4 if self == ShipType.CRUISE else 3

                psd[index] += k - 20*math.log10(f_1/f_ref) \
                                - 10*math.log10((1 - f/f_1)**2 + d**2)

        return frequencies, psd

    def get_speed_range(self) -> typing.Tuple[lps_qty.Speed, lps_qty.Speed]:
        """ Return a Speed in the expected range by ship type.

        http://1worldenergy.com/ship-sizes-speeds-voyage-times-maritime-regulations/
        """
        speed_ranges = {
            ShipType.BULKER: (10, 15),
            ShipType.CONTAINERSHIP: (18, 25),
            ShipType.CRUISE: (20, 24),
            ShipType.DREDGER: (4, 12),
            ShipType.FISHING: (8, 12),
            ShipType.GOVERNMENT: (12, 20),
            ShipType.RESEARCH: (12, 20),
            ShipType.NAVAL: (25, 30),
            ShipType.PASSENGER: (15, 22),
            ShipType.RECREATIONAL: (5, 20),
            ShipType.TANKER: (12, 16),
            ShipType.TUG: (10, 14),
            ShipType.VEHICLE_CARRIER: (15, 19),
            ShipType.OTHER: (5, 25),
        }
        return lps_qty.Speed.kt(speed_ranges[self][0]), lps_qty.Speed.kt(speed_ranges[self][1])

    def get_rpm_range(self) -> typing.Tuple[lps_qty.Frequency, lps_qty.Frequency]:
        """ Return an RPM range in the expected range by ship type. """
        rpm_ranges = {
            ShipType.BULKER: (50, 90),
            ShipType.CONTAINERSHIP: (60, 120),
            ShipType.CRUISE: (80, 110),
            ShipType.DREDGER: (30, 60),
            ShipType.FISHING: (50, 90),
            ShipType.GOVERNMENT: (70, 110),
            ShipType.RESEARCH: (70, 110),
            ShipType.NAVAL: (100, 180),
            ShipType.PASSENGER: (60, 100),
            ShipType.RECREATIONAL: (40, 90),
            ShipType.TANKER: (50, 80),
            ShipType.TUG: (100, 150),
            ShipType.VEHICLE_CARRIER: (60, 90),
            ShipType.OTHER: (40, 150),
        }
        return lps_qty.Frequency.rpm(rpm_ranges[self][0]), \
                lps_qty.Frequency.rpm(rpm_ranges[self][1])

    def get_random_speed(self) -> lps_qty.Speed:
        """ Return a Speed in the expected range by ship type.

        http://1worldenergy.com/ship-sizes-speeds-voyage-times-maritime-regulations/
        """
        min_speed, max_speed = self.get_speed_range()
        return lps_qty.Speed.kt(random.uniform(min_speed.get_kt(), max_speed.get_kt()))

    def get_random_length(self) -> lps_qty.Distance:
        """ Return a Length in the expected range by ship type. """
        length_ranges = {
            ShipType.BULKER: (150, 300),
            ShipType.CONTAINERSHIP: (200, 400),
            ShipType.CRUISE: (250, 350),
            ShipType.DREDGER: (50, 150),
            ShipType.FISHING: (30, 100),
            ShipType.GOVERNMENT: (50, 120),
            ShipType.RESEARCH: (50, 120),
            ShipType.NAVAL: (100, 250),
            ShipType.PASSENGER: (100, 200),
            ShipType.RECREATIONAL: (10, 50),
            ShipType.TANKER: (200, 350),
            ShipType.TUG: (20, 40),
            ShipType.VEHICLE_CARRIER: (150, 200),
            ShipType.OTHER: (10, 400),
        }
        return lps_qty.Distance.m(random.uniform(*length_ranges[self]))

    def get_random_draft(self) -> lps_qty.Distance:
        """ Return a Draft (calado) in the expected range by ship type. """
        draft_ranges = {
            ShipType.BULKER: (10, 18),
            ShipType.CONTAINERSHIP: (12, 15),
            ShipType.CRUISE: (7, 9),
            ShipType.DREDGER: (5, 8),
            ShipType.FISHING: (3, 6),
            ShipType.GOVERNMENT: (5, 9),
            ShipType.RESEARCH: (5, 9),
            ShipType.NAVAL: (7, 12),
            ShipType.PASSENGER: (6, 10),
            ShipType.RECREATIONAL: (1, 3),
            ShipType.TANKER: (12, 20),
            ShipType.TUG: (3, 5),
            ShipType.VEHICLE_CARRIER: (8, 12),
            ShipType.OTHER: (3, 12),
        }
        return lps_qty.Distance.m(random.uniform(*draft_ranges[self]))

    def get_blades_range(self) -> typing.Tuple[int, int]:
        """ Return a range of valores típicos de pás para cada tipo de navio. """
        blades_ranges = {
            ShipType.BULKER: (4, 5),
            ShipType.CONTAINERSHIP: (4, 6),
            ShipType.CRUISE: (5, 6),
            ShipType.DREDGER: (3, 4),
            ShipType.FISHING: (3, 4),
            ShipType.GOVERNMENT: (4, 5),
            ShipType.RESEARCH: (4, 5),
            ShipType.NAVAL: (5, 7),
            ShipType.PASSENGER: (4, 6),
            ShipType.RECREATIONAL: (3, 5),
            ShipType.TANKER: (4, 5),
            ShipType.TUG: (3, 4),
            ShipType.VEHICLE_CARRIER: (4, 5),
            ShipType.OTHER: (3, 6)
        }
        return blades_ranges[self]

    def get_shafts_range(self) -> typing.Tuple[int, int]:
        """ Return a range of shafts typical for each ship type. """
        shafts_ranges = {
            ShipType.BULKER: (1, 2),
            ShipType.CONTAINERSHIP: (1, 2),
            ShipType.CRUISE: (2, 3),
            ShipType.DREDGER: (1, 2),
            ShipType.FISHING: (1, 1),
            ShipType.GOVERNMENT: (1, 2),
            ShipType.RESEARCH: (1, 2),
            ShipType.NAVAL: (2, 2),
            ShipType.PASSENGER: (1, 2),
            ShipType.RECREATIONAL: (1, 2),
            ShipType.TANKER: (1, 2),
            ShipType.TUG: (1, 2),
            ShipType.VEHICLE_CARRIER: (1, 2),
            ShipType.OTHER: (1, 2),
        }
        return shafts_ranges[self]


class Propulsion():
    """ Class to simulate ship propulsion system noise modulation. """

    def __init__(self,
                 ship_type:
                 ShipType,
                 n_blades: int,
                 n_shafts: int,
                 shaft_error=5e-2,
                 blade_error=1e-3):
        self.ship_type = ship_type
        self.n_blades = n_blades
        self.n_shafts = n_shafts
        self.shaft_error = shaft_error
        self.blade_error = blade_error

    def estimate_rpm(self, speeds: typing.List[lps_qty.Speed]) -> typing.List[lps_qty.Frequency]:
        """
        Estimate RPM based on ship speed.

        Args:
            speeds: List of speeds in lps_qty.Speed.

        Returns:
            List of estimated RPM (lps_qty.Frequency).
        """
        rpm_estimates = []
        for speed in speeds:
            if speed == lps_qty.Speed.m_s(0):
                rpm_estimates.append(lps_qty.Frequency.rpm(0))
                return

            if speed < lps_qty.Speed.m_s(0):
                speed *= -1

            min_rpm, max_rpm = self.ship_type.get_rpm_range()
            _, max_speed = self.ship_type.get_speed_range()

            rpm_value = min_rpm + (max_rpm - min_rpm) * (speed / max_speed)
            rpm_estimates.append(rpm_value)

        return rpm_estimates

    def estimate_modulation(self, rpms: typing.List[lps_qty.Frequency]) -> typing.List[float]:
        """
        Estimate modulation index based on RPM.

        Args:
            rpms: List of RPM (lps_qty.Frequency).

        Returns:
            List of modulation indices (0 to 1).
        """
        modulation_indices = []
        for rpm in rpms:
            min_rpm, max_rpm = self.ship_type.get_rpm_range()
            cavitation_threshold = min_rpm * 1.15

            modulation_index = (rpm - cavitation_threshold) / (max_rpm - cavitation_threshold)
            modulation_index = np.clip(modulation_index, 0, 1)
            modulation_indices.append(modulation_index)

        return modulation_indices

    def modulate_noise(self,
                        broadband: np.array,
                        speeds: typing.List[lps_qty.Speed],
                        fs = lps_qty.Frequency):
        """
        Modulate broadband noise based on ship speeds.

        Args:
            broadband: Array of broadband noise samples.
            speeds: List of speed values at corresponding timestamps.

        Returns:
            Array of modulated noise.
        """

        n_samples = len(broadband)

        if n_samples != len(speeds):
            speed_list = np.array([s.get_m_s() for s in speeds])
            speeds_interp = np.interp(np.linspace(0, 1, n_samples),
                              np.linspace(0, 1, len(speeds)),
                              speed_list)
            speeds = [lps_qty.Speed.m_s(s) for s in speeds_interp]

        rpms = self.estimate_rpm(speeds)
        modulation_indices = self.estimate_modulation(rpms)

        narrowband_total = np.zeros(n_samples)

        for _ in range(self.n_shafts):

            base_error = self.shaft_error * np.random.randn()
            along_error = 0.1 * np.random.randn(n_samples)

            shaft_rpm = np.array([rpm.get_hz() * (1 + base_error) for rpm in rpms]) + along_error

            shaft_phase = np.cumsum(shaft_rpm) / fs.get_hz()

            sins = np.zeros((self.n_blades, n_samples))
            for har in range(1, self.n_blades + 1):
                blade_phase_shift = (har - 1) * 2 * np.pi / self.n_blades
                harmonic_variation = 1 + self.blade_error * np.random.randn()
                sins[har - 1, :] = np.cos(2 * np.pi * shaft_phase * har * harmonic_variation + \
                                         blade_phase_shift)

            narrowband_eixo = np.sum(sins, axis=0)
            narrowband_total += narrowband_eixo

        narrowband_total /= self.n_shafts * self.n_blades

        modulated_signal = (1 + np.array(modulation_indices) * narrowband_total) * broadband

        return modulated_signal, narrowband_total

    @classmethod
    def get_random(cls, ship_type: ShipType) -> 'Propulsion':
        """ Return a random propulsion based on ship type. """
        return cls(ship_type = ship_type,
                          n_blades = np.random.randint(*ship_type.get_blades_range()),
                          n_shafts = np.random.randint(*ship_type.get_shafts_range()))

class Ship(lps_dynamic.Element):
    """ Class to represent a Ship in the scenario"""

    def __init__(self,
                 ship_id: str,
                 ship_type: ShipType,
                 max_speed: lps_qty.Speed = None,
                 length: lps_qty.Distance = None,
                 draft: lps_qty.Distance = None,
                 propulsion: Propulsion = None,
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> None:

        self.ship_id = ship_id
        self.ship_type = ship_type
        self.length = length if length is not None else ship_type.get_random_length()
        self.draft = draft if draft is not None else ship_type.get_random_draft()
        self.propulsion = propulsion if propulsion is not None else Propulsion.get_random(ship_type)
        initial_state.max_speed = max_speed if max_speed is not None else \
            ship_type.get_random_speed()

        super().__init__(initial_state=initial_state)

    def generate_base_noise(self, fs: lps_qty.Frequency) -> np.array:
        """
        Generates ship noise over simulated intervals.

        Args:
            fs: The sampling frequency as an lps_qty.Frequency.

        Returns:
            A numpy array containing the audio signal in 1 µPa @1m.
        """

        audio_signals = []
        speeds = []

        for state, interval in zip(self.state_map, self.step_interval):
            speeds.append(state.velocity.get_magnitude())
            frequencies, psd = self.ship_type.to_psd(fs=fs,
                                                     lenght=self.length,
                                                     speed=speeds[-1])

            freqs_hz = [f.get_hz() for f in frequencies]

            audio_signals.append(lps_bb.generate(frequencies=np.array(freqs_hz),
                                                 psd_db=psd,
                                                 n_samples=int(interval * fs),
                                                 fs=fs.get_hz()))

        return np.concatenate(audio_signals), speeds

    def generate_noise(self, fs: lps_qty.Frequency) -> np.array:
        """ Generate noise based on simulated steps. """
        noise, speeds = self.generate_base_noise(fs=fs)
        noise, _ = self.propulsion.modulate_noise(broadband=noise, speeds=speeds, fs=fs)
        return noise

    @overrides.overrides
    def get_depth(self) -> lps_qty.Distance:
        """ Return the starting depth of the element. """
        return self.draft

class Scenario():
    """ Class to represent a Scenario """

    def __init__(self,
                 environment: lps_env.Environment = lps_env.Environment.random(),
                 channel_desc: lps_desc.Description = lps_desc.Description.get_random(),
                 temp_dir: str = '/temp') \
                    -> None:
        self.environment = environment
        self.channel_desc = channel_desc
        self.channel = None
        self.sonars = {}
        self.ships = {}
        self.n_steps = 0
        self.step_interval = None
        self.temp_dir = temp_dir

    def add_sonar(self, sonar_id: str, sonar: lps_sonar.Sonar) -> None:
        """ Insert a sonar in the scenario. """
        self.sonars[sonar_id] = sonar

    def add_ship(self, ship: Ship) -> None:
        """ Insert a ship in the scenario. """
        self.ships[ship.ship_id] = ship

    def reset(self) -> None:
        """ Reset simulation. """
        self.n_steps = 0
        self.step_interval = None

        for _, ship in self.ships.items():
            ship.reset()

        for _, sonar in self.sonars.items():
            sonar.reset()

    def simulate(self, step_interval: lps_qty.Speed, n_steps: int = 1) -> None:
        """Move elements n times the step_interval

        Args:
            step_interval (lps_qty.Speed): Interval of a step
            n_steps (int, optional): Number of steps. Defaults to 1.
        """

        self.n_steps = n_steps
        self.step_interval = step_interval

        for _, ship in self.ships.items():
            ship.move(step_interval=step_interval, n_steps=n_steps)

        for _, sonar in self.sonars.items():
            sonar.move(step_interval=step_interval, n_steps=n_steps)

    def geographic_plot(self, filename: str) -> None:
        """ Make plots with top view, centered in the final position of the sonar. """

        def plot_ship(x, y, angle, ship_id):
            plt.scatter(x[-1], y[-1], marker=(3, 0, angle - 90), s=200)
            plt.plot(x, y, label=f'{ship_id}')

        for sonar_id, sonar in self.sonars.items():

            _, ax = plt.subplots(figsize=(8, 8))
            limit = 0

            ref_x = []
            ref_y = []
            ref_angle = 0
            for step_i in range(self.n_steps):
                diff = sonar[step_i].position
                diff_vel = sonar[step_i].velocity

                ref_x.append(diff.x.get_km())
                ref_y.append(diff.y.get_km())
                ref_angle = diff_vel.get_azimuth().get_deg()

            for ship_id, ship in self.ships.items():

                x = []
                y = []
                last_angle = 0
                for step_i in range(self.n_steps):
                    diff = ship[step_i].position
                    diff_vel = ship[step_i].velocity

                    x.append(diff.x.get_km() - ref_x[-1])
                    y.append(diff.y.get_km() - ref_y[-1])
                    last_angle = diff_vel.get_azimuth().get_deg()


                limit = np.max([limit, np.max(np.abs(x)), np.max(np.abs(y))])
                plot_ship(x, y, last_angle, ship_id)

            ref_x = np.array(ref_x)
            ref_y = np.array(ref_y)
            limit = np.max([limit,
                            np.max(np.abs(ref_x - ref_x[-1])),
                            np.max(np.abs(ref_y - ref_y[-1]))])
            plot_ship(ref_x - ref_x[-1], ref_y - ref_y[-1], ref_angle, "Sonar")

            plt.xlabel('X (km)')
            plt.ylabel('Y (km)')
            plt.legend()

            limit *= 1.2
            plt.xlim(-limit, limit)
            plt.ylim(-limit, limit)
            ax.set_aspect('equal')

            if len(self.sonars) == 1:
                output_filename = filename
            else:
                output_filename = f"{filename}{sonar_id}.png"
            plt.savefig(output_filename)

            plt.clf()
        plt.close()

    def relative_distance_plot(self, filename: str) -> None:
        """ Make plots with relative distances between each sonar and ships. """

        for sonar_id, sonar in self.sonars.items():

            t = [(step_i * self.step_interval).get_s() for step_i in range(self.n_steps)]

            for ship_id, ship in self.ships.items():

                dist = []
                for step_i in range(self.n_steps):
                    diff = ship[step_i].position - sonar[step_i].position
                    dist.append(diff.get_magnitude_xy().get_km())

                plt.plot(t, dist, label=ship_id)

            plt.xlabel('Time (seconds)')
            plt.ylabel('Distance (km)')
            plt.legend()

            if len(self.sonars) == 1:
                output_filename = filename
            else:
                output_filename = f"{filename}{sonar_id}.png"
            plt.savefig(output_filename)

            plt.clf()
        plt.close()

    def get_sonar_audio(self, sonar_id: str, fs: lps_qty.Frequency):
        """ Returns the calculated scan data for the selected sonar. """

        sonar = self.sonars[sonar_id]

        ship_signals = {}
        source_depths = {}

        for ship_id, ship in tqdm.tqdm(self.ships.items(),
                                       desc="Generating noise from ships",
                                       leave=False):
            ship_signals[ship_id] = ship.generate_noise(fs=fs)
            source_depths[ship_id] = ship.get_depth()

        sonar_signals = []
        for sensor in sonar.sensors:

            ship_distance = {}
            max_distance = lps_qty.Distance.m(0)

            for ship_id, ship in tqdm.tqdm(self.ships.items(),
                                        desc="Generating noise from ships",
                                        leave=False):

                ship_distance[ship_id] = []
                for step_id in range(self.n_steps):
                    sonar_pos = sonar[step_id].position + sensor.rel_position
                    dist = (ship[step_id].position - sonar_pos).get_magnitude()
                    ship_distance[ship_id].append(dist)
                    if ship_distance[ship_id][-1] > max_distance:
                        max_distance = ship_distance[ship_id][-1]

            max_distance = lps_qty.Distance.km(np.ceil(max_distance.get_km()*2)/2)

            channel = lps_channel.Channel(
                    description = self.channel_desc,
                    source_depths = [
                        lps_qty.Distance.m(5),
                        lps_qty.Distance.m(7),
                        lps_qty.Distance.m(25)
                    ],
                    sensor_depth = sonar.get_depth(),
                    max_distance = max_distance,
                    max_distance_points = 200,
                    sample_frequency = fs,
                    temp_dir = self.temp_dir)

            for ship_id in tqdm.tqdm(self.ships.keys(),
                                    desc="Propagating signals from ships",
                                    leave=False):
                ship_signals[ship_id] = channel.propagate(input_data = ship_signals[ship_id],
                                                        source_depth = source_depths[ship_id],
                                                        distance = ship_distance[ship_id])

            ship_signal = np.sum(np.column_stack(list(ship_signals.values())), axis=1)
            env_noise = self.environment.generate_bg_noise(len(ship_signal), fs=fs.get_hz())

            sonar_signals.append(sensor.apply(ship_signal + env_noise))

        return np.column_stack(sonar_signals)
