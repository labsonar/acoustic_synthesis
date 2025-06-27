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
import lps_synthesis.propagation.models as lps_model


class Scenario():
    """ Class to represent a Scenario """

    def __init__(self,
            step_interval: lps_qty.Time,
            channel: lps_channel.Channel,
            environment: lps_env.Environment = lps_env.Environment.random()) \
                    -> None:
        self.environment = environment
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

                    x.append(diff.x.get_km() - ref_x[-1])
                    y.append(diff.y.get_km() - ref_y[-1])
                    last_angle = diff_vel.get_azimuth().get_deg()


                limit = np.max([limit, np.max(np.abs(x)), np.max(np.abs(y))])
                plot_ship(x, y, last_angle, container.get_id())

            ref_x = np.array(ref_x)
            ref_y = np.array(ref_y)
            limit = np.max([limit,
                            np.max(np.abs(ref_x - ref_x[-1])),
                            np.max(np.abs(ref_y - ref_y[-1]))])
            plot_ship(ref_x - ref_x[-1], ref_y - ref_y[-1], ref_angle, "Sonar", 200)

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

                plt.plot(t, dist, label=container.get_id())

            plt.xlabel('Time (seconds)')
            plt.ylabel('Distance (km)')
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

            plt.plot(t, speeds, label=container.get_id())

        plt.xlabel('Time (second)')
        plt.ylabel('Speed (knot)')
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

                plt.plot(t, speeds, label=container.get_id())

                # for source in container.noise_sources:

                #     speeds = []
                #     for step_i in range(self.n_steps):
                #         speeds.append((source[step_i].get_relative_speed(sonar[step_i])).get_kt())

                #     plt.plot(t, speeds, label=source.get_id())

            plt.xlabel('Time (second)')
            plt.ylabel('Speed (knot)')
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

    @staticmethod
    def _process_noise_source(noise_source, fs):
        source_id = noise_source.get_id()
        noise = noise_source.generate_noise(fs=fs)
        depth = noise_source.get_depth()
        return source_id, noise, depth, noise_source

    def _calculate_sensor_signal(self,
                                 sensor,
                                 sonar,
                                 source_ids,
                                 noises_dict,
                                 depth_dict,
                                 noise_dict,
                                 fs,
                                 channel,
                                 environment,
                                 n_steps):
        distance_dict = {}
        gain_dict = {}

        for container in self.noise_containers:
            for noise_source in container.noise_sources:
                distance_dict[noise_source.get_id()] = [
                    (noise_source[step_id].position - sensor[step_id].position).get_magnitude()
                    for step_id in range(n_steps)
                ]
                gain_dict[noise_source.get_id()] = [
                    sensor.direction_gain(step_id, noise_source[step_id].position)
                    for step_id in range(n_steps)
                ]

        noises = []

        for source_id in tqdm.tqdm(source_ids,
                                desc="Propagating signals from sources",
                                leave=False,
                                ncols=120):
            sound_speed = channel.description.get_speed_at(depth_dict[source_id])

            source_doppler_list = [
                noise_dict[source_id][step_i].get_relative_speed(sonar[step_i])
                for step_i in range(n_steps)
            ]
            sensor_doppler_list = [
                sonar[step_i].get_relative_speed(noise_dict[source_id][step_i])
                for step_i in range(n_steps)
            ]

            gain = scipy.resample(gain_dict[source_id], len(noises_dict[source_id]))

            doppler_noise = lps_model.apply_doppler(
                input_data=noises_dict[source_id] * gain,
                speeds=source_doppler_list,
                sound_speed=sound_speed
            )

            propag_noise = channel.propagate(
                input_data=doppler_noise,
                source_depth=depth_dict[source_id],
                distance=distance_dict[source_id]
            )

            noises.append(lps_model.apply_doppler(
                input_data=propag_noise,
                speeds=sensor_doppler_list,
                sound_speed=sound_speed
            ))

        min_size = min(signal.shape[0] for signal in noises)
        signals = [signal[:min_size] for signal in noises]
        ship_signal = np.sum(np.column_stack(signals), axis=1)
        env_noise = environment.generate_bg_noise(len(ship_signal), fs=fs.get_hz())

        return sensor.apply(ship_signal + env_noise)

    def get_sonar_audio(self, sonar_id: str, fs: lps_qty.Frequency):
        """ Returns the calculated scan data for the selected sonar. """

        print(f"##### Getting sonar audio for {sonar_id} sonar #####")
        sonar = self.sonars[sonar_id]

        source_ids = []
        noises_dict = {}
        depth_dict = {}
        noise_dict = {}

        with future_lib.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(Scenario._process_noise_source, noise_source, fs)
                for container in self.noise_containers
                for noise_source in container.noise_sources
            ]

            for future in tqdm.tqdm(future_lib.as_completed(futures), total=len(futures),
                                    desc="Noise Sources", leave=False, ncols=120):
                source_id, noise, depth, noise_source = future.result()
                source_ids.append(source_id)
                noises_dict[source_id] = noise
                depth_dict[source_id] = depth
                noise_dict[source_id] = noise_source

        print(f"##### Audio for {len(source_ids)} sources generated #####")

        sonar_signals = []
        with future_lib.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    self._calculate_sensor_signal,
                    sensor, sonar, source_ids, noises_dict, depth_dict,
                    noise_dict, fs, self.channel, self.environment, self.n_steps
                )
                for sensor in sonar.sensors
            ]

            for future in tqdm.tqdm(future_lib.as_completed(futures), total=len(futures),
                                    desc="Sensors", leave=False, ncols=120):
                sonar_signals.append(future.result())

        min_size = min(signal.shape[0] for signal in sonar_signals)
        signals = [signal[:min_size] for signal in sonar_signals]
        signals = np.column_stack(signals)
        print(f"##### Audio compiled totalizing {signals.shape} samples #####")

        return signals
