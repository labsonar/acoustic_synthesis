"""
Module for representing the scenario and their elements
"""
import enum
import typing
import math
import random

import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.acoustical_channel as lps_channel

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

    def get_random_speed(self) -> lps_qty.Speed:
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
        return lps_qty.Speed.kt(random.uniform(*speed_ranges[self]))

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

class Ship(lps_dynamic.Element):
    """ Class to represent a Ship in the scenario"""

    def __init__(self,
                 ship_id: str,
                 ship_type: ShipType,
                 max_speed: lps_qty.Speed = None,
                 length: lps_qty.Distance = None,
                 draft: lps_qty.Distance = None,
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> None:

        self.ship_id = ship_id
        self.ship_type = ship_type
        self.length = length if length is not None else ship_type.get_random_length()
        self.draft = draft if draft is not None else ship_type.get_random_draft()
        initial_state.max_speed = max_speed if max_speed is not None else \
            ship_type.get_random_speed()

        super().__init__(initial_state=initial_state)

class Scenario():
    """ Class to represent a Scenario """

    def __init__(self,
                 environment: lps_env.Environment = lps_env.Environment.random(),
                 channel_desc: lps_channel.Description
                 start_time = lps_qty.Timestamp()) -> None:
        self.environment = environment
        self.start_time = start_time
        self.sonars = {}
        self.ships = {}
        self.times = None

    def add_sonar(self, sonar_id: str, sonar: lps_sonar.Sonar) -> None:
        """ Insert a sonar in the scenario. """
        self.sonars[sonar_id] = sonar

    def add_ship(self, ship: Ship) -> None:
        """ Insert a ship in the scenario. """
        self.ships[ship.ship_id] = ship

    def simulate(self, time_step: lps_qty.Time, simulation_time: lps_qty.Time) -> \
                                                                    typing.List[lps_qty.Timestamp]:
        """ Calculate the positions of all elements in the scenario along the simulation time. """

        self.times = [self.start_time + lps_qty.Time.s(t) for t in np.linspace(0,
                                simulation_time.get_s(),
                                int(simulation_time.get_s()//time_step.get_s()))]

        for _, ship in self.ships.items():
            ship.reset(self.start_time)
            ship.move(self.times)

        for _, sonar in self.sonars.items():
            sonar.reset(self.start_time)
            sonar.move(self.times)

        return self.times

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
            for time in self.times:
                diff = sonar[time].position
                diff_vel = sonar[time].velocity

                ref_x.append(diff.x.get_km())
                ref_y.append(diff.y.get_km())
                ref_angle = diff_vel.get_azimuth().get_deg()

            for ship_id, ship in self.ships.items():

                x = []
                y = []
                last_angle = 0
                for time in self.times:
                    diff = ship[time].position
                    diff_vel = ship[time].velocity

                    x.append(diff.x.get_km() - ref_x[-1])
                    y.append(diff.y.get_km() - ref_y[-1])
                    last_angle = diff_vel.get_azimuth().get_deg()


                limit = np.max([limit, np.max(np.abs(x)), np.max(np.abs(y))])
                plot_ship(x, y, last_angle, ship_id)

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

            t = [(time - self.times[0]).get_s() for time in self.times]

            for ship_id, ship in self.ships.items():

                dist = []
                for time in self.times:
                    diff = ship[time].position - sonar[time].position
                    diff.z = lps_qty.Distance.m(0)

                    dist.append(diff.get_magnitude().get_km())

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
