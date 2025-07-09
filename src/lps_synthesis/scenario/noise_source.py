"""
Module for representing the scenario and their elements
"""
import os
import enum
import typing
import math
import random
import abc
import concurrent.futures as future_lib

import tqdm
import overrides
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_signal
import lps_synthesis.scenario.dynamic as lps_dynamic
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

class NoiseCompiler():
    """ Class to compile and group multiple noise sources based on their spatial reference.

    Using:
        __iter__():
            Iterates over compiled noise groups, yielding (signal, depth, source list) tuples.
    """

    def __init__(self, noise_containers: typing.List[NoiseContainer], fs: lps_qty.Frequency):
        self.keys = []
        self.signal_dict = {}
        self.depth_dict = {}
        self.source_list_dict = {}
        self.fs = fs

        with future_lib.ThreadPoolExecutor(max_workers=None) as executor:
            futures = [
                executor.submit(NoiseCompiler._process_noise_source, noise_source, fs)
                for container in noise_containers
                for noise_source in container.noise_sources
            ]

            for future in tqdm.tqdm(future_lib.as_completed(futures), total=len(futures),
                                    desc="Compiling noise sources", leave=False, ncols=120):
                noise, depth, noise_source, key = future.result()

                if key not in self.keys:
                    self.keys.append(key)
                    self.signal_dict[key] = noise
                    self.depth_dict[key] = depth
                    self.source_list_dict[key] = [noise_source]
                else:
                    self.signal_dict[key] += noise
                    self.source_list_dict[key].append(noise_source)

    def __iter__(self) -> typing.Iterator[typing.Tuple[np.ndarray,
                                                       lps_qty.Distance,
                                                       typing.List[NoiseSource]]]:
        """Iterate over (signal, depth, source_list) triplets."""
        for key in self.keys:
            yield (self.signal_dict[key],
                   self.depth_dict[key],
                   self.source_list_dict[key])

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def _process_noise_source(noise_source: NoiseSource, fs: lps_qty.Frequency):
        noise = noise_source.generate_noise(fs=fs)
        depth = noise_source.get_depth()
        key = (id(noise_source.ref_element), str(noise_source.rel_position))
        return noise, depth, noise_source, key

    def save_plot(self, filename: str) -> None:
        """Save a plot with the PSD of all noise signals in the compiler."""

        plt.figure(figsize=(12, 6))

        for i, (signal, _, _) in enumerate(self):

            freqs, psd_vals = lps_bb.psd(
                signal=signal,
                fs=self.fs.get_hz()
            )

            plt.plot(freqs, psd_vals, label=f"Signal {i}")

        plt.title("PSD of All Noise Signals")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("dB re 1μPa²/Hz")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def show_details(self) -> str:
        """ Show the noise in the compiler. """

        for signal, depth, source_list in self:

            ids = [src.get_id() for src in source_list]

            print(f" - Sources: {ids}")
            print(f"\tDepth: {depth}, N samples: {len(signal)}")

    def save_wavs(self, base_dir: str) -> None:
        """ Save all noise as .wav. """

        for i, (signal, _, _) in enumerate(self):
            lps_signal.save_normalized_wav(signal,
                                           self.fs,
                                           os.path.join(base_dir, f"source_{i}.wav"))



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

        speed_factor = speed/v_c
        lenght_factor = lenght/l_0
        psd = np.zeros(len(frequencies))

        if speed_factor == 0 or lenght_factor == 0:
            return psd, frequencies

        psd = psd + 60*math.log10(speed_factor) + 20*math.log10(lenght_factor)

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

    def draw_max_speed(self, seed: int) -> lps_qty.Speed:
        """ Return a max speed in the expected range by ship type. """
        rng = random.Random(seed)
        value = rng.uniform(1.2, 1.8)
        return self.draw_cruising_speed(seed) * value

    def draw_cruising_speed(self, seed: int) -> lps_qty.Speed:
        """ Return a cruising speed in the expected range by ship type. """
        speed_range = {
            ShipType.BULKER: (11, 15),
            ShipType.CONTAINERSHIP: (18, 25),
            ShipType.CRUISE: (20, 24),
            ShipType.DREDGER: (6, 10),
            ShipType.FISHING: (8, 14),
            ShipType.GOVERNMENT: (12, 18),
            ShipType.RESEARCH: (11, 16),
            ShipType.NAVAL: (18, 30),
            ShipType.PASSENGER: (15, 22),
            ShipType.RECREATIONAL: (10, 35),
            ShipType.TANKER: (10, 15),
            ShipType.TUG: (10, 14),
            ShipType.VEHICLE_CARRIER: (16, 20),
            ShipType.OTHER: (5, 25),
        }
        rng = random.Random(seed)
        value = rng.uniform(*(speed_range[self]))*10//1/10
        return lps_qty.Speed.kt(value)

    def draw_cruising_rotacional_frequency(self, seed: int) -> lps_qty.Frequency:
        """ Return a cruising rotacional frequency in the expected range by ship type. """
        rpm_range = {
            ShipType.BULKER: (60, 100),
            ShipType.CONTAINERSHIP: (80, 110),
            ShipType.CRUISE: (90, 130),
            ShipType.DREDGER: (80, 180),
            ShipType.FISHING: (200, 600),
            ShipType.GOVERNMENT: (100, 300),
            ShipType.RESEARCH: (100, 300),
            ShipType.NAVAL: (120, 300),
            ShipType.PASSENGER: (100, 180),
            ShipType.RECREATIONAL: (200, 500),
            ShipType.TANKER: (60, 100),
            ShipType.TUG: (200, 600),
            ShipType.VEHICLE_CARRIER: (70, 110),
            ShipType.OTHER: (60, 600),
        }
        rng = random.Random(seed)
        value = rng.uniform(*(rpm_range[self]))*10//1/10
        return lps_qty.Frequency.rpm(value)

    def draw_length(self, seed: int) -> lps_qty.Distance:
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
        rng = random.Random(seed)
        value = rng.uniform(*(length_ranges[self]))*10//1/10
        return lps_qty.Distance.m(value)

    def draw_draft(self, seed: int) -> lps_qty.Distance:
        """ Return a Draft in the expected range by ship type. """
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
        rng = random.Random(seed)
        value = rng.randint(*(draft_ranges[self]))
        return lps_qty.Distance.m(value)

    def draw_n_blades(self, seed: int) -> int:
        """ Return the number of blades of a ship in the expected range by ship type. """
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
        rng = random.Random(seed)
        value = rng.randint(*(blades_ranges[self]))
        return value

    def draw_n_shafts(self, seed: int) -> typing.Tuple[int, int]:
        """ Return the number of shafts of a ship in the expected range by ship type. """
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
        rng = random.Random(seed)
        value = rng.randint(*(shafts_ranges[self]))
        return value

class CavitationNoise(NoiseSource):
    """ Class to simulate ship propulsion system noise modulation. """

    def __init__(self,
                 ship_type: ShipType,
                 n_blades: int = None,
                 n_shafts: int = None,
                 length: lps_qty.Distance = None,
                 cruise_speed: lps_qty.Speed = None,
                 cruise_rotacional_frequency: lps_qty.Frequency = None,
                 max_speed: lps_qty.Speed = None,
                 rotacional_coeficient: lps_qty.Distance = None, # expected to be around 5,34 m
                 cavitation_threshold: lps_qty.Frequency = None,
                 seed : int = None
                 ):
        self.seed = seed if seed is not None else id(self)
        self.ship_type = ship_type
        self.n_blades = n_blades if n_blades is not None else \
                        ship_type.draw_n_blades(self.seed)
        self.n_shafts = n_shafts if n_shafts is not None else \
                        ship_type.draw_n_shafts(self.seed)
        self.length = length if length is not None else \
                        ship_type.draw_length(self.seed)
        self.cruise_speed = cruise_speed if cruise_speed is not None else \
                        ship_type.draw_cruising_speed(self.seed)
        self.cruise_rotacional_frequency = cruise_rotacional_frequency \
                        if cruise_rotacional_frequency is not None else \
                        ship_type.draw_cruising_rotacional_frequency(self.seed)
        self.max_speed = max_speed if max_speed is not None else \
                        ship_type.draw_max_speed(self.seed)
        self.rotacional_coeficient = rotacional_coeficient \
                        if rotacional_coeficient is not None else \
                        self._draw_rotacional_coeficient(self.seed)

        rng = random.Random(self.seed)
        self.cavitation_threshold = cavitation_threshold if cavitation_threshold is not None else \
                        self.cruise_rotacional_frequency * rng.uniform(0.2,0.6)

        super().__init__(source_id="cavitation")

    def _draw_rotacional_coeficient(self, seed : int) -> lps_qty.Distance:
        rng = random.Random(seed)
        value = 0.089 * (1 + rng.uniform(-0.2, 0.2))
        # constante original calculada com m min/s aplicação em m
        return lps_qty.Distance.m(value * 60)

    def _draw_harmonic_intensities(self,
                                   mod_ind : float,
                                   n_harmonics : float = None,
                                   blade_gain : float = None,
                                   decay : float = None,
                                   harmonic_std : float = None)-> \
                                        typing.Tuple[float, np.array]:

        rng = np.random.default_rng(seed = self.seed)
        n_harmonics = n_harmonics if n_harmonics is not None else \
                        (rng.integers(3,6) * self.n_blades)
        blade_gain = blade_gain if blade_gain is not None else rng.uniform(2, 2.5)
        decay = decay if decay is not None else rng.uniform(0.4,1)
        harmonic_std = harmonic_std if harmonic_std is not None else rng.uniform(0.05,0.2)

        if mod_ind < 1e-3:
            return 1, np.zeros(n_harmonics)

        n_k = rng.normal(1, harmonic_std, self.n_blades)
        decays = decay + rng.normal(0, decay * 0.01, self.n_blades)

        intensities = []
        for n in range(1, n_harmonics + 1):

            if n % self.n_blades == 0:
                base_intensity = blade_gain * n_k[n%self.n_blades] / \
                        ((n//self.n_blades) ** decays[n%self.n_blades])
            else:
                base_intensity = n_k[n%self.n_blades] / \
                        (np.ceil(n/self.n_blades) ** decays[n%self.n_blades])

            intensity = base_intensity
            intensities.append(np.max(intensity,0))

        intensities = np.array(intensities)
        a0 =  np.sum(intensities)/mod_ind

        total_energy = a0**2 + np.sum(intensities**2)/2

        a0 /= np.sqrt(total_energy)
        intensities /= np.sqrt(total_energy)

        return a0, np.array(intensities)

    def _check_rotacional_coeficient(self, speeds):
        non_zero_speeds = [abs(s) for s in speeds if s != lps_qty.Speed.kt(0)]
        if not non_zero_speeds:
            return False

        min_speed = min(non_zero_speeds)
        min_coef = (min_speed - self.cruise_speed)/\
                    (lps_qty.Frequency.rpm(15) - self.cruise_rotacional_frequency)

        if self.rotacional_coeficient < min_coef:
            self.rotacional_coeficient = min_coef
        return True

    def estimate_rpm(self, speeds: typing.List[lps_qty.Speed]) -> typing.List[lps_qty.Frequency]:
        """
        Estimate RPM based on ship speed.

        Args:
            speeds: List of speeds in lps_qty.Speed.

        Returns:
            List of estimated RPM (lps_qty.Frequency).
        """
        if self._check_rotacional_coeficient(speeds):
            return [(abs(s) - self.cruise_speed)/self.rotacional_coeficient +
                        self.cruise_rotacional_frequency for s in speeds]
        else:
            return [lps_qty.Frequency.rpm(0)] * len(speeds)

    def estimate_modulation(self, speeds: typing.List[lps_qty.Speed]) -> typing.List[float]:
        """
        Estimate modulation index based on ship speed and cavitation threshold.

        Args:
            speeds: List of ship speeds (lps_qty.Speed).

        Returns:
            List of modulation indices (from 0 to 1).
        """
        modulation_indices = []
        rng = np.random.default_rng(self.seed)
        base_mod_index = rng.uniform(0.3, 0.5)
        cruise_mod_index = rng.uniform(0.6, 0.8)

        for speed in speeds:
            aux = abs(speed)

            if aux < self.cavitation_threshold:
                modulation_index = 0.0
            else:

                progress = ((aux - self.cavitation_threshold) /
                            (self.cruise_rotacional_frequency - self.cavitation_threshold))

                modulation_index = base_mod_index + progress * (cruise_mod_index - base_mod_index)
                modulation_index = np.clip(modulation_index, 0, 1)

            modulation_indices.append(modulation_index)

        return modulation_indices

    def modulate_noise(self,
                        broadband: np.array,
                        speeds: typing.List[lps_qty.Speed],
                        fs = lps_qty.Frequency) -> typing.Tuple[np.array, np.array]:
        """
        Modulate broadband noise based on ship speeds.

        Args:
            broadband: Array of broadband noise samples.
            speeds: List of speed values at corresponding timestamps.

        Returns:
            Array of modulated noise.
            Array of modulating noise.
        """

        n_samples = len(broadband)

        rpms = self.estimate_rpm(speeds)
        rpms = np.array([rpm.get_rpm() for rpm in rpms])

        if n_samples != len(rpms):
            rpms = np.interp(np.linspace(0, 1, n_samples),
                              np.linspace(0, 1, len(speeds)),
                              rpms)

        discrete_rpms = np.round(rpms).astype(int)
        rpms_values = np.unique(discrete_rpms)
        mod_inds = self.estimate_modulation([lps_qty.Frequency.rpm(r) for r in rpms_values])

        rpm_to_harmonics = {}
        for rpm_val, mod_ind in zip(rpms_values, mod_inds):
            a0, an = self._draw_harmonic_intensities(mod_ind=mod_ind)
            rpm_to_harmonics[rpm_val] = (a0, an)

        delta_t = 1 / fs.get_rpm()
        phase_accum = 2 * np.pi * np.cumsum(rpms) * delta_t

        narrowband = np.zeros(n_samples)
        for i in range(n_samples):
            a0_i, an_i = rpm_to_harmonics[discrete_rpms[i]]

            narrowband[i] += a0_i
            for n, a in enumerate(an_i):
                narrowband[i] += a * np.cos(phase_accum[i] * (1+n))

        modulated_signal = narrowband * broadband
        return modulated_signal, narrowband

    @classmethod
    def get_random(cls, ship_type: ShipType = None, seed : int = None) -> 'CavitationNoise':
        """ Return a random propulsion based on ship type. """
        if ship_type is None:
            rng = random.Random(seed)
            return cls.get_random(rng.choice(list(ShipType)))
        return cls(ship_type = ship_type, seed = seed)

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
        filter_state = None

        rng = np.random.default_rng(seed = self.seed)

        for state, interval in self.ref_element.get_simulated_steps():
            speeds.append(state.velocity.get_magnitude())
            frequencies, psd = self.ship_type.to_psd(fs=fs,
                                                     lenght=self.length,
                                                     speed=speeds[-1])

            freqs_hz = [f.get_hz() for f in frequencies]

            noise, filter_state = lps_bb.generate(frequencies=np.array(freqs_hz),
                                                 psd_db=psd,
                                                 n_samples=int(interval * fs),
                                                 fs=fs.get_hz(),
                                                 filter_state=filter_state,
                                                 seed=rng)
            audio_signals.append(noise)

        return np.concatenate(audio_signals), speeds

    @overrides.overrides
    def generate_noise(self, fs: lps_qty.Frequency) -> np.array:
        """ Generate noise based on simulated steps. """
        broadband, speeds = self.generate_broadband_noise(fs=fs)
        modulated_noise, _ = self.modulate_noise(broadband=broadband, speeds=speeds, fs=fs)
        return modulated_noise

class NarrowBandNoise(NoiseSource):
    """
    Narrowband noise source of the form:
        V(t) = (A * (1 + ε(t))) * cos(2π f0 t + φ(t))

    JAN LI... “Review of PM and AM Noise Measurement Systems” doi: 10.1109/ICMMT.1998.768259.
    ε(t) -> disturbance amplitude in %
    φ(t) -> between 0 and 2π

    Note:
        Default case, simple tonal noise
    """

    def __init__(self,
                 frequency: lps_qty.Frequency,
                 amp_db_p_upa: float,
                 epsilon_fn: typing.Callable[[np.ndarray], np.ndarray] = np.zeros_like,
                 phi_fn: typing.Callable[[np.ndarray], np.ndarray] = np.zeros_like,
                 rel_position: lps_dynamic.Displacement =
                        lps_dynamic.Displacement(lps_qty.Distance.m(0), lps_qty.Distance.m(0))):
        super().__init__(source_id=f"NarrowBand [{frequency}]", rel_position=rel_position)
        self.frequency = frequency
        self.amp = 10 ** (amp_db_p_upa / 20)
        self.epsilon_fn = epsilon_fn
        self.phi_fn = phi_fn

    def generate_noise(self, fs: lps_qty.Frequency) -> np.ndarray:
        accum_interval = lps_qty.Time.s(0)
        for _, interval in self.ref_element.get_simulated_steps():
            accum_interval = accum_interval + interval

        n_samples = int(accum_interval * fs)
        t = np.linspace(0, accum_interval.get_s(), n_samples, endpoint=False)

        amplitude = self.amp * (1+self.epsilon_fn(t))
        phase = self.phi_fn(t)

        return amplitude * np.cos(2 * np.pi * self.frequency.get_hz() * t + phase)

    @classmethod
    def with_sine_am_modulation(cls,
                                frequency: lps_qty.Frequency,
                                amp_db_p_upa: float,
                                am_freq: lps_qty.Frequency,
                                am_depth: float = 0.1,
                                rel_position: lps_dynamic.Displacement =
                                        lps_dynamic.Displacement(lps_qty.Distance.m(0),
                                                                 lps_qty.Distance.m(0))):
        """
        Creates a NarrowBandNoise source with sinusoidal amplitude modulation (AM).

        The generated signal is:
            V(t) = A * [1 + m * sin(2π f_m t)] * cos(2π f_c t)

        Where:
            - f_c = carrier frequency (`frequency`)
            - f_m = modulation frequency (`am_freq`)
            - m   = modulation depth (`am_depth`, 0 ≤ m ≤ 1)
            - A   = amplitude corresponding to `amp_db_p_upa`
        """
        return cls(frequency,
                   amp_db_p_upa,
                   lambda t: am_depth * np.sin(2 * np.pi * am_freq.get_hz() * t),
                   np.zeros_like,
                   rel_position)

    @classmethod
    def with_sine_fm_modulation(cls,
                        frequency: lps_qty.Frequency,
                        amp_db_p_upa: float,
                        oscilation_freq: lps_qty.Frequency,
                        deviation_freq: lps_qty.Frequency,
                        rel_position: lps_dynamic.Displacement = lps_dynamic.Displacement(lps_qty.Distance.m(0), lps_qty.Distance.m(0))):
        """
        Creates a NarrowBandNoise source with sinusoidal frequency modulation (FM).

        The generated signal is:
            V(t) = A * cos(2π f_c t + β * sin(2π f_m t))

        Where:
            - f_c = carrier frequency (`frequency`)
            - f_m = modulation frequency (`oscilation_freq`)
            - β   = frequency deviation / modulation frequency = Δf / f_m
            - Δf  = peak frequency deviation (`deviation_freq`)
            - A   = amplitude corresponding to `amp_db_p_upa`
        """
        return cls(frequency,
                   amp_db_p_upa,
                   np.zeros_like,
                   lambda t: deviation_freq/oscilation_freq * np.sin(2 * np.pi * oscilation_freq.get_hz() * t),
                   rel_position)

    @classmethod
    def with_fm_chirp(cls,
                        amp_db_p_upa: float,
                        start_frequency: lps_qty.Frequency,
                        end_frequency: lps_qty.Frequency,
                        tx_interval: lps_qty.Time,
                        tx_duration: lps_qty.Time,
                        rel_position: lps_dynamic.Displacement = lps_dynamic.Displacement(lps_qty.Distance.m(0), lps_qty.Distance.m(0))):
        """
        Creates a NarrowBandNoise source with linear frequency modulated chirp.

        The generated signal is:
            V(t) = A * cos(2π (f₀ t + ½ k t²)), for 0 ≤ t < tx_duration
            V(t) = 0, otherwise (within each tx_interval period)

        Where:
            - f₀ = start_frequency
            - f₁ = end_frequency
            - k  = (f₁ - f₀) / tx_duration
            - A  = amplitude corresponding to `amp_db_p_upa`
            - The signal repeats every `tx_interval` seconds,
            and is silent during the pause (`tx_interval - tx_duration`)
        """

        def phi_fn(t: np.ndarray) -> np.ndarray:
            k = (end_frequency - start_frequency).get_hz() / tx_duration.get_s()
            t_mod = np.mod(t, tx_interval.get_s())
            phase = np.zeros_like(t)

            active = t_mod < tx_duration.get_s()
            t_active = t_mod[active]

            phase[active] = 2 * np.pi * (0.5 * k * t_active ** 2)
            return phase

        def epsilon_fn(t: np.ndarray) -> np.ndarray:
            t_mod = np.mod(t, tx_interval.get_s())
            active = t_mod < tx_duration.get_s()
            return np.where(active, 0.0, -1.0)

        return cls(start_frequency,
                   amp_db_p_upa,
                   epsilon_fn,
                   phi_fn,
                   rel_position)

    @classmethod
    def with_brownian_modulation(cls,
                                frequency: lps_qty.Frequency,
                                amp_db_p_upa: float,
                                amp_std: float = 0.02,
                                phase_std: float = 0.02,
                                seed: int = None,
                                rel_position: lps_dynamic.Displacement =
                                    lps_dynamic.Displacement(lps_qty.Distance.m(0),
                                                            lps_qty.Distance.m(0))):
        """
        Creates a NarrowBandNoise source with Brownian (random walk) modulation
        applied independently to amplitude and phase.

        The generated signal is:
            V(t) = A * [1 + ε(t)] * cos(2π f_c t + φ(t))

        Where:
            - f_c = carrier frequency (`frequency`)
            - ε(t): cumulative sum of Gaussian noise (amplitude modulation)
            - φ(t): cumulative sum of Gaussian noise (phase modulation)
            - A = amplitude corresponding to `amp_db_p_upa`

        Notes:
            - The randomness is seeded for reproducibility (`seed`)
            - The `amp_std` controls the standard deviation of each step in ε(t)
            - The `phase_std` controls the same for φ(t)
        """
        rng = np.random.default_rng(seed)

        def make_brownian(scale: float) -> typing.Callable[[np.ndarray], np.ndarray]:
            def brownian_fn(t: np.ndarray) -> np.ndarray:
                steps = rng.normal(scale=scale, size=len(t))
                return np.cumsum(steps)
            return brownian_fn

        epsilon_fn = make_brownian(amp_std)
        phi_fn = make_brownian(phase_std)

        return cls(frequency, amp_db_p_upa, epsilon_fn, phi_fn, rel_position)


class Ship(NoiseContainer):
    """ Class to represent a Ship in the scenario"""

    def __init__(self,
                 ship_id: str,
                 propulsion: CavitationNoise = None,
                 draft: lps_qty.Distance = None,
                 initial_state: lps_dynamic.State = None,
                 seed: int = None) -> None:

        self.seed = seed if seed is not None else id(self)

        self.propulsion = propulsion if propulsion is not None else \
                            CavitationNoise.get_random(self.seed)
        self.ship_type = propulsion.ship_type
        self.draft = draft if draft is not None else self.ship_type.draw_draft(self.seed)

        if initial_state is None:
            initial_state = lps_dynamic.State()
            initial_state.velocity.x = propulsion.cruise_speed
            initial_state.position.x = -1 * lps_qty.Time.s(5) * propulsion.cruise_speed

        initial_state.max_speed = propulsion.max_speed

        super().__init__(container_id=ship_id, initial_state=initial_state)
        self.add_source(propulsion)

    @overrides.overrides
    def get_depth(self) -> lps_qty.Distance:
        """ Return the starting depth of the element. """
        return self.draft

    @classmethod
    def by_type(cls, ship_type = ShipType, seed: int = None) -> 'Ship':
        """ Instanciate a Ship based on type. """
        return Ship(ship_id=str(ship_type), propulsion=CavitationNoise(ship_type, seed=seed), seed=seed)
