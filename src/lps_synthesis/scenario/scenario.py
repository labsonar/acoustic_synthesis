"""
Module for representing the scenario and their elements
"""
import enum
import typing

import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic

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

class Ship(lps_dynamic.Element):
    """ Class to represent a Ship in the scenario """

    def __init__(self,
                 ship_id: str,
                 ship_type: ShipType,
                 time: lps_qty.Timestamp = lps_qty.Timestamp(),
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> None:
        super().__init__(time, initial_state)
        self.ship_id = ship_id
        self.ship_type = ship_type

class AcousticSensor():
    """ Class to represent an AcousticSensor in the scenario """

    def __init__(self,
                 sensitivity: lps_qty.Sensitivity = None,
                 rel_position: lps_dynamic.Displacement = \
                        lps_dynamic.Displacement(lps_qty.Distance.m(0), lps_qty.Distance.m(0))
                 ) -> None:
        self.sensitivity = sensitivity
        self.rel_position = rel_position

class Sonar(lps_dynamic.Element):
    """ Class to represent a Sonar (with multiple acoustic sensors) in the scenario """

    def __init__(self,
                 sensors: typing.List[AcousticSensor],
                 time: lps_qty.Timestamp = lps_qty.Timestamp(),
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> None:
        super().__init__(time, initial_state)
        self.sensors = sensors

    @staticmethod
    def hidrofone(
                 sensitivity: lps_qty.Sensitivity = None,
                 time: lps_qty.Timestamp = lps_qty.Timestamp(),
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> 'Sonar':
        """ Class constructor for construct a Sonar with only one sensor """
        return Sonar(sensors = [AcousticSensor(sensitivity=sensitivity)],
                     time=time, initial_state=initial_state)

class Scenario():
    """ Class to represent a Scenario """

    def __init__(self, start_time = lps_qty.Timestamp()) -> None:
        self.start_time = start_time
        self.sonars = {}
        self.ships = {}
        self.times = None

    def add_sonar(self, sonar_id: str, sonar: Sonar) -> None:
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
            ship.move(self.times)

        for _, sonar in self.sonars.items():
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
