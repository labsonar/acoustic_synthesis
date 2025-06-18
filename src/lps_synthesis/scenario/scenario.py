"""
Module for representing the scenario and their elements
"""
import enum
import typing
import math
import random
import overrides
import abc
import time

import tqdm
import tikzplotlib as tikz
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scipy

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.channel as lps_channel
import lps_synthesis.propagation.models as lps_model
import lps_sp.acoustical.broadband as lps_bb

class NoiseSource(lps_dynamic.RelativeElement):
    """ Abstract class to represent a point source of noise. """

    def __init__(self, source_id: str, rel_position: lps_dynamic.Displacement = \
                 lps_dynamic.Displacement(lps_qty.Distance.m(0), lps_qty.Distance.m(0))):
        self.source_id = source_id
        super().__init__(rel_position=rel_position)

    @abc.abstractmethod
    def generate_noise(self, fs: lps_qty.Frequency) -> np.array:
        """ Generate noise based on simulated steps. """

    def get_id(self) -> str:
        """ Return the relative id """
        if self.ref_element is None or not isinstance(self.ref_element, NoiseContainer):
            return self.source_id
        return f"{self.ref_element.container_id}:{self.source_id}"

class NoiseContainer(lps_dynamic.Element):
    """ Base class to represent an element with multiple point noise sources. """

    def __init__(self, container_id: str, initial_state: lps_dynamic.State = lps_dynamic.State()):
        super().__init__(initial_state)
        self.container_id = container_id
        self.noise_sources = []

    def add_source(self, noise_source: NoiseSource):
        """ Add a noise source relative to this container element. """
        self.noise_sources.append(noise_source)
        noise_source.set_base_element(self)

    def get_id(self) -> str:
        """ Return the relative id """
        if len(self.noise_sources) == 1:
            return self.container_id
        return f"{self.container_id}[{len(self.noise_sources)}]"


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

        # Converto to source level
        # for index, f in enumerate(frequencies):
        #     psd[index] += 10*math.log10(0.231 * f.get_hz())

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

class CavitationNoise(NoiseSource):
    """ Class to simulate ship propulsion system noise modulation. """

    def __init__(self,
                 ship_type: ShipType,
                 n_blades: int,
                 n_shafts: int,
                 shaft_error = 5e-2,
                 blade_error = 1e-3,
                 length: lps_qty.Distance = None):
        self.ship_type = ship_type
        self.n_blades = n_blades
        self.n_shafts = n_shafts
        self.shaft_error = shaft_error
        self.blade_error = blade_error
        self.length = length if length is not None else ship_type.get_random_length()
        super().__init__(source_id="cavitation")

    @staticmethod
    def generate_harmonic_intensities(n_blades, n_harmonics=16, alpha=1.5, gain=1.5, noise_std=0.1, mod_ind = 2):

        intensities = []
        rn = np.random.normal(1, noise_std, n_blades)
        alphas = alpha + np.random.normal(0, alpha * 0.01, n_blades)
        for n in range(1, n_harmonics + 1):

            # Reforço para múltiplos de k
            if n % n_blades == 0:
                base_intensity = gain * rn[n%n_blades] / ((n//n_blades) ** alphas[n%n_blades])
            else:
                base_intensity = rn[n%n_blades] / (np.ceil(n/n_blades) ** alphas[n%n_blades])

            # Adiciona ruído leve
            intensity = base_intensity
            intensities.append(np.max(intensity,0))

        intensities = np.array(intensities)
        A0 =  np.sum(intensities)/max(mod_ind, 1e-2)

        total_energy = A0**2 + np.sum(intensities**2)/2

        A0 /= np.sqrt(total_energy)
        intensities /= np.sqrt(total_energy)

        return A0, np.array(intensities)

    def estimate_rpm(self, speeds: typing.List[lps_qty.Speed]) -> typing.List[lps_qty.Frequency]:
        """
        Estimate RPM based on ship speed.

        Args:
            speeds: List of speeds in lps_qty.Speed.

        Returns:
            List of estimated RPM (lps_qty.Frequency).
        """
        rpm_estimates = []

        min_rpm, max_rpm = self.ship_type.get_rpm_range()
        min_speed, max_speed = self.ship_type.get_speed_range()

        last_speed = lps_qty.Speed.m_s(0)
        rpm_value = lps_qty.Frequency.rpm(0)

        for speed in speeds:
            if last_speed != speed:
                if speed == lps_qty.Speed.m_s(0):
                    rpm_value = lps_qty.Frequency.rpm(0)

                else:
                    if speed < lps_qty.Speed.m_s(0):
                        speed *= -1

                    # if min_speed > speed:
                    #     rpm_value = min_rpm
                    # else:
                    #     rpm_value = min_rpm + (max_rpm - min_rpm) * ((speed-min_speed) / (max_speed-min_speed))

                    rpm_value = min_rpm + (max_rpm - min_rpm) * (speed / max_speed)

            rpm_estimates.append(rpm_value)
            last_speed = speed

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
        min_rpm, max_rpm = self.ship_type.get_rpm_range()

        for rpm in rpms:
            cavitation_threshold = lps_qty.Frequency.rpm(40)

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

        rpms = self.estimate_rpm(speeds)
        rpms = np.array([rpm.get_hz() for rpm in rpms])

        if n_samples != len(rpms):
            rpms = np.interp(np.linspace(0, 1, n_samples),
                              np.linspace(0, 1, len(speeds)),
                              rpms)

        discrete_rpms = np.round(rpms).astype(int)
        rpms_values = np.unique(discrete_rpms)
        mod_inds = self.estimate_modulation([lps_qty.Frequency.hz(r) for r in rpms_values])

        rpm_to_harmonics = {}
        for rpm_val, mod_ind in zip(rpms_values, mod_inds):
            A0, A = CavitationNoise.generate_harmonic_intensities(
                n_blades=self.n_blades, mod_ind=mod_ind)
            rpm_to_harmonics[rpm_val] = (A0, A)


        delta_t = 1 / fs.get_hz()
        phase_accum = 2 * np.pi * np.cumsum(rpms) * delta_t

        narrowband = np.zeros(n_samples)
        for i in range(n_samples):
            A0_i, A_i = rpm_to_harmonics[discrete_rpms[i]]

            narrowband[i] += A0_i
            for n, An in enumerate(A_i):
                narrowband[i] += An * np.cos(phase_accum[i] * (1+n))

        # A0, A = CavitationNoise.generate_harmonic_intensities(n_blades=self.n_blades, mod_ind=0.8) #TODO couple mod_ind to estimate_modulation

        # delta_t = 1 / fs.get_hz()

        # narrowband = np.ones(n_samples) * A0

        # for n, An in enumerate(A):
        #     phase = 2 * np.pi * (1 + n) * np.cumsum(rpms) * delta_t
        #     narrowband += An * np.cos(phase)

        modulated_signal = narrowband * broadband
        return modulated_signal, narrowband

    @classmethod
    def get_random(cls, ship_type: ShipType = None) -> 'CavitationNoise':
        """ Return a random propulsion based on ship type. """
        if ship_type is None:
            return cls.get_random(random.choice(list(ShipType)))
        return cls(ship_type = ship_type,
                          n_blades = np.random.randint(*ship_type.get_blades_range()),
                          n_shafts = np.random.randint(*ship_type.get_shafts_range()))

    def generate_broadband_noise(self, fs: lps_qty.Frequency) -> np.array:
        """
        Generates ship noise over simulated intervals.

        Args:
            fs: The sampling frequency as an lps_qty.Frequency.

        Returns:
            A numpy array containing the audio signal in 1 µPa @1m.
        """
        self.check()

        audio_signals = []
        speeds = []

        for state, interval in self.ref_element.get_simulated_steps():
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

    @overrides.overrides
    def generate_noise(self, fs: lps_qty.Frequency) -> np.array:
        """ Generate noise based on simulated steps. """
        broadband, speeds = self.generate_broadband_noise(fs=fs)
        modulated_noise, _ = self.modulate_noise(broadband=broadband, speeds=speeds, fs=fs)
        return modulated_noise

class Sin(NoiseSource):
    """ Simple noise source that add a sin. """

    def __init__(self,
                 frequency: lps_qty.Frequency,
                 amp_db_p_upa: float,
                 rel_position: lps_dynamic.Displacement = \
                         lps_dynamic.Displacement(lps_qty.Distance.m(0), lps_qty.Distance.m(0))):
        super().__init__(source_id=f"Sin [{frequency}]", rel_position=rel_position)
        self.frequency = frequency
        self.amp = 10**(amp_db_p_upa/20)

    @abc.abstractmethod
    def generate_noise(self, fs: lps_qty.Frequency) -> np.array:
        """ Generate noise based on simulated steps. """
        accum_interval = lps_qty.Time.s(0)
        for _, interval in self.ref_element.get_simulated_steps():
            accum_interval = accum_interval + interval

        n_samples = int(accum_interval * fs)
        t = np.linspace(0, accum_interval.get_s(), n_samples, endpoint=False)

        return np.sin(2 * np.pi * self.frequency.get_hz() * t) * self.amp

class Ship(NoiseContainer):
    """ Class to represent a Ship in the scenario"""

    def __init__(self,
                 ship_id: str,
                 propulsion: CavitationNoise = None,
                 max_speed: lps_qty.Speed = None,
                 draft: lps_qty.Distance = None,
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> None:

        self.propulsion = propulsion if propulsion is not None else CavitationNoise.get_random()
        self.ship_type = propulsion.ship_type
        self.draft = draft if draft is not None else self.ship_type.get_random_draft()
        initial_state.max_speed = max_speed if max_speed is not None else \
                                                            self.ship_type.get_random_speed()

        super().__init__(container_id=ship_id, initial_state=initial_state)
        self.add_source(propulsion)


    @overrides.overrides
    def get_depth(self) -> lps_qty.Distance:
        """ Return the starting depth of the element. """
        return self.draft


class Scenario():
    """ Class to represent a Scenario """

    def __init__(self,
            channel: lps_channel.Channel,
            environment: lps_env.Environment = lps_env.Environment.random()) \
                    -> None:
        self.environment = environment
        self.channel = channel
        self.sonars = {}
        self.noise_containers = []
        self.n_steps = 0
        self.step_interval = None

    def add_sonar(self, sonar_id: str, sonar: lps_sonar.Sonar) -> None:
        """ Insert a sonar in the scenario. """
        self.sonars[sonar_id] = sonar

    def add_noise_source(self, noise_source: NoiseSource, initial_state: lps_dynamic.State) -> None:
        """ Insert a noise_source in the scenario. """
        container = NoiseContainer(container_id="", initial_state=initial_state)
        container.add_source(noise_source=noise_source)
        self.add_noise_container(container)

    def add_noise_container(self, noise_source: NoiseContainer) -> None:
        """ Insert a noise container in the scenario. """
        self.noise_containers.append(noise_source)

    def reset(self) -> None:
        """ Reset simulation. """
        self.n_steps = 0
        self.step_interval = None

        for container in self.noise_containers:
            container.reset()

        for _, sonar in self.sonars.items():
            sonar.reset()

    def simulate(self, step_interval: lps_qty.Speed, n_steps: int = 1) -> None:
        """Move elements n times the step_interval

        Args:
            step_interval (lps_qty.Speed): Interval of a step
            n_steps (int, optional): Number of steps. Defaults to 1.
        """

        self.n_steps += n_steps
        self.step_interval = step_interval

        for container in self.noise_containers:
            container.move(step_interval=step_interval, n_steps=n_steps)

        for _, sonar in self.sonars.items():
            sonar.move(step_interval=step_interval, n_steps=n_steps)

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

    def _calculate_sensor_signal(self, sensor, sonar, source_ids, noises_dict, depth_dict, noise_dict, fs, channel, environment, n_steps):
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

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(Scenario._process_noise_source, noise_source, fs)
                for container in self.noise_containers
                for noise_source in container.noise_sources
            ]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Noise Sources", leave=False, ncols=120):
                source_id, noise, depth, noise_source = future.result()
                source_ids.append(source_id)
                noises_dict[source_id] = noise
                depth_dict[source_id] = depth
                noise_dict[source_id] = noise_source

        print(f"##### Audio for {len(source_ids)} sources generated #####")

        # Parallelize sensor signal calculations
        sonar_signals = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(
                    self._calculate_sensor_signal,
                    sensor, sonar, source_ids, noises_dict, depth_dict, noise_dict, fs, self.channel, self.environment, self.n_steps
                )
                for sensor in sonar.sensors
            ]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Sensors", leave=False, ncols=120):
                sonar_signals.append(future.result())

        # Align all signals to the minimum length
        min_size = min(signal.shape[0] for signal in sonar_signals)
        signals = [signal[:min_size] for signal in sonar_signals]
        signals = np.column_stack(signals)
        print(f"##### Audio compiled totalizing {signals.shape} samples #####")

        return signals
