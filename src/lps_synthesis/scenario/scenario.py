"""
Module for representing the scenario and their elements
"""
import concurrent.futures as future_lib
import tqdm
import tikzplotlib as tikz

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scipy

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.scenario.noise_source as lps_noise
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.channel as lps_channel


class Scenario():
    """ Class to represent a Scenario """

    def __init__(self,
            step_interval: lps_qty.Time,
            channel: lps_channel.Channel,
            environment: lps_env.Environment | None = None) -> None:
        self.environment = environment or lps_env.Environment.random()
        self.channel = channel
        self.sonars = {}
        self.noise_containers = []
        self.n_steps = 0
        self.step_interval = step_interval

    def add_sonar(self, sonar_id: str, sonar: lps_sonar.Sonar) -> None:
        """ Insert a sonar in the scenario. """
        self.sonars[sonar_id] = sonar

    def add_noise_source(self,
                         noise_source: lps_noise.NoiseSource,
                         initial_state: lps_dynamic.State) -> None:
        """ Insert a noise_source in the scenario. """
        container = lps_noise.NoiseContainer(container_id="", initial_state=initial_state)
        container.add_source(noise_source=noise_source)
        self.add_noise_container(container)

    def add_noise_container(self, noise_source: lps_noise.NoiseContainer) -> None:
        """ Insert a noise container in the scenario. """
        self.noise_containers.append(noise_source)

    def reset(self, step_interval: lps_qty.Time = None) -> None:
        """ Reset simulation. """
        self.n_steps = 0
        self.step_interval = step_interval if step_interval is not None else self.step_interval

        for container in self.noise_containers:
            container.reset()

        for _, sonar in self.sonars.items():
            sonar.reset()

    def simulate(self, n_steps: int = 1) -> None:
        """Move elements n times the step_interval

        Args:
            step_interval (lps_qty.Speed): Interval of a step
            n_steps (int, optional): Number of steps. Defaults to 1.
        """

        self.n_steps += n_steps

        for container in self.noise_containers:
            container.move(step_interval=self.step_interval, n_steps=n_steps)

        for _, sonar in self.sonars.items():
            sonar.move(step_interval=self.step_interval, n_steps=n_steps)

    def geographic_plot(self, filename: str) -> None:
        """ Make plots with top view, centered in the final position of the sonar. """

        def plot_ship(x, y, angle, ship_id, s = 100):
            plt.scatter(x[-1], y[-1], marker=(3, 0, angle - 90), s=s)
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

            for container in self.noise_containers:

                x = []
                y = []
                last_angle = 0
                for step_i in range(self.n_steps):
                    diff = container[step_i].position
                    diff_vel = container[step_i].velocity

                    # x.append(diff.x.get_km() - ref_x[0])
                    # y.append(diff.y.get_km() - ref_y[0])
                    x.append(diff.x.get_km())
                    y.append(diff.y.get_km())
                    last_angle = diff_vel.get_azimuth().get_deg()


                limit = np.max([limit, np.max(np.abs(x)), np.max(np.abs(y))])
                plot_ship(x, y, last_angle, container.get_id())

            ref_x = np.array(ref_x)
            ref_y = np.array(ref_y)
            limit = np.max([limit,
                            np.max(np.abs(ref_x)),
                            np.max(np.abs(ref_y))])
                            # np.max(np.abs(ref_x - ref_x[0])),
                            # np.max(np.abs(ref_y - ref_y[0]))])
            # plot_ship(ref_x - ref_x[0], ref_y - ref_y[0], ref_angle, "Sonar", 200)
            plot_ship(ref_x, ref_y, ref_angle, "Sonar", 200)

            # cont = 0
            # for sensor in sonar.sensors:

            #     x = []
            #     y = []
            #     for step_i in range(self.n_steps):
            #         pos = sensor[step_i].position

            #         x.append(pos.x.get_km() - ref_x[-1])
            #         y.append(pos.y.get_km() - ref_y[-1])


            #     limit = np.max([limit, np.max(np.abs(x)), np.max(np.abs(y))])
            #     plot_ship(x, y, 0, f"Sensor {cont}")
            #     cont += 1

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

            if output_filename[-4:] == ".tex":
                tikz.save(output_filename)
            else:
                plt.savefig(output_filename)

            plt.clf()
        plt.close()

    def relative_distance_plot(self, filename: str) -> None:
        """ Make plots with relative distances between each sonar and ships. """

        for sonar_id, sonar in self.sonars.items():

            t = [(step_i * self.step_interval).get_s() for step_i in range(self.n_steps)]

            for container in self.noise_containers:

                dist = []
                for step_i in range(self.n_steps):
                    diff = container[step_i].position - sonar[step_i].position
                    dist.append(diff.get_magnitude_xy().get_km())

                # plt.plot(t, dist, label=container.get_id())
                plt.plot(dist, t, label=container.get_id())

            plt.ylabel('Time (seconds)')
            plt.xlabel('Distance (km)')
            plt.gca().invert_yaxis()
            # plt.xlabel('Time (seconds)')
            # plt.ylabel('Distance (km)')
            plt.legend()

            if len(self.sonars) == 1:
                output_filename = filename
            else:
                output_filename = f"{filename}{sonar_id}.png"

            if output_filename[-4:] == ".tex":
                tikz.save(output_filename)
            else:
                plt.savefig(output_filename)

            plt.clf()
        plt.close()

    def velocity_plot(self, filename: str) -> None:
        """ Make plots with velocity of all noise sources. """

        t = [(step_i * self.step_interval).get_s() for step_i in range(self.n_steps)]

        for container in self.noise_containers:

            speeds = []
            for step_i in range(self.n_steps):
                speeds.append(container[step_i].velocity.get_magnitude_xy().get_kt())

            # plt.plot(t, speeds, label=container.get_id())
            plt.plot(speeds, t, label=container.get_id())

        plt.ylabel('Time (second)')
        plt.xlabel('Speed (knot)')
        plt.gca().invert_yaxis()
        # plt.xlabel('Time (second)')
        # plt.ylabel('Speed (knot)')
        plt.legend()

        output_filename = filename
        if output_filename[-4:] == ".tex":
            tikz.save(output_filename)
        else:
            plt.savefig(output_filename)
        plt.close()

    def relative_velocity_plot(self, filename: str) -> None:
        """ Make plots with relative distances between each sonar and ships. """

        for sonar_id, sonar in self.sonars.items():

            t = [(step_i * self.step_interval).get_s() for step_i in range(self.n_steps)]

            for container in self.noise_containers:

                speeds = []
                for step_i in range(self.n_steps):
                    speeds.append((container[step_i].get_relative_speed(sonar[step_i])).get_kt())

                # plt.plot(t, speeds, label=container.get_id())

                # for source in container.noise_sources:

                #     speeds = []
                #     for step_i in range(self.n_steps):
                #         speeds.append((source[step_i].get_relative_speed(sonar[step_i])).get_kt())

                #     plt.plot(t, speeds, label=source.get_id())

                plt.plot(speeds, t, label=container.get_id())
            plt.ylabel('Time (second)')
            plt.xlabel('Speed (knot)')
            plt.gca().invert_yaxis()
            # plt.xlabel('Time (second)')
            # plt.ylabel('Speed (knot)')
            plt.legend()

            if len(self.sonars) == 1:
                output_filename = filename
            else:
                output_filename = f"{filename}{sonar_id}.png"

            if output_filename[-4:] == ".tex":
                tikz.save(output_filename)
            else:
                plt.savefig(output_filename)

            plt.clf()
        plt.close()

    def bearing_plot(self, filename: str) -> None:
        """ Make plots with absolute bearing. """

        for sonar_id, sonar in self.sonars.items():

            t = [(step_i * self.step_interval).get_s() for step_i in range(self.n_steps)]

            for container in self.noise_containers:

                angle = []
                for step_i in range(self.n_steps):
                    angle.append((container[step_i].position - sonar[step_i].position).get_azimuth().get_ncw_deg())

                plt.plot(angle, t, label=container.get_id())

            angle = []
            for step_i in range(self.n_steps):
                angle.append(sonar[step_i].velocity.get_azimuth().get_ncw_deg())

            plt.plot(angle, t, label=sonar_id)


            plt.ylabel('Time (second)')
            plt.xlabel('Bearing (degree)')
            plt.gca().invert_yaxis()
            # plt.xlabel('Time (second)')
            # plt.ylabel('Speed (knot)')
            plt.legend()

            if len(self.sonars) == 1:
                output_filename = filename
            else:
                output_filename = f"{filename}{sonar_id}.png"

            if output_filename[-4:] == ".tex":
                tikz.save(output_filename)
            else:
                plt.savefig(output_filename)

            plt.clf()
        plt.close()

    def relative_bearing_plot(self, filename: str) -> None:
        """ Make plots with relative bearing. """

        for sonar_id, sonar in self.sonars.items():

            t = [(step_i * self.step_interval).get_s() for step_i in range(self.n_steps)]

            for container in self.noise_containers:

                angle = []
                for step_i in range(self.n_steps):
                    abs_angle = (container[step_i].position - sonar[step_i].position).get_azimuth()
                    ref_angle = sonar[step_i].velocity.get_azimuth()
                    rel_angle = abs_angle - ref_angle
                    angle.append(rel_angle.get_ccw_deg())

                plt.plot(angle, t, label=container.get_id())

            plt.ylabel('Time (second)')
            plt.xlabel('Bearing (degree)')
            plt.gca().invert_yaxis()
            plt.legend()

            if len(self.sonars) == 1:
                output_filename = filename
            else:
                output_filename = f"{filename}{sonar_id}.png"

            if output_filename[-4:] == ".tex":
                tikz.save(output_filename)
            else:
                plt.savefig(output_filename)

            plt.clf()
        plt.close()

    def get_sonar_audio(self, sonar_id: str, fs: lps_qty.Frequency):
        """ Returns the calculated scan data for the selected sonar. """
        sonar = self.sonars[sonar_id]
        compiler = lps_noise.NoiseCompiler(self.noise_containers, fs=fs)
        return sonar.get_data(compiler, self.channel, self.environment)
