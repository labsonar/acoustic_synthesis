""" Sonar Module. """
import typing
import math
import abc
import os
import concurrent.futures as future_lib

import tqdm
import overrides
import numpy as np
import scipy.signal as scipy
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.noise_source as lps_noise
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.channel as lps_channel
import lps_synthesis.propagation.models as lps_model

class Directivity():
    """ Class to represent the gain of a polar diagram for an acoustic sensor. """
    @abc.abstractmethod
    def get_gain(self, relative_angle: lps_qty.RelativeBearing):
        """ return the directional gain for an specific angle. """

class Omnidirectional(Directivity):
    """ Class to represent an acoustic omnidirectional sensor. """
    @overrides.overrides
    def get_gain(self, relative_angle: lps_qty.RelativeBearing):
        """ return the directional gain for an specific angle. """
        return 1

class Shaded(Directivity):
    """ Class to represent an acoustic omnidirectional sensor.
    https://www.coe.ufrj.br/~fabriciomtb/papers/Estima%C3%A7%C3%A3o_da_Dire%C3%A7%C3%A3o_de_Chegada_Utilizando_MUSIC_com_Antenas_Direcionais_em_Arranjo_Circular.pdf
    """
    def __init__(self, d: float = 10, m: float = 8.7):
        super().__init__()
        self.d = d
        self.m = m

    @overrides.overrides
    def get_gain(self, relative_angle: lps_qty.RelativeBearing):
        """ return the directional gain for an specific angle. """
        max_gain = math.sqrt((self.d / 2**self.m) * ((1 + 1)**self.m))
        gain = math.sqrt((self.d/2**self.m) *
                         ((1 + math.cos(relative_angle.get_ccw_rad()))**self.m))
        return gain/max_gain


class Sensitivity():
    """ Class to represent the sensitivity of an acoustic sensor. """
    @abc.abstractmethod
    def convert(self, input_data: np.array, fs: lps_qty.Frequency) -> np.array:
        """ convert an input signal in µPa para um sinal em V. """

class FlatBand(Sensitivity):
    """Represents a sensor with flat sensitivity over frequency (band-independent)."""

    def __init__(self, sensitivity: lps_qty.Sensitivity):
        super().__init__()
        self.sensitivity = sensitivity

    @overrides.overrides
    def convert(self, input_data: np.array, fs: lps_qty.Frequency) -> np.array:
        """ convert an input signal in µPa para um sinal em V. """
        return input_data * 10**((self.sensitivity.get_db_v_p_upa()) / 20)


class AcousticSensor(lps_dynamic.RelativeElement):
    """ Class to represent an AcousticSensor in the scenario """

    def __init__(self,
                 sensitivity: Sensitivity = FlatBand(lps_qty.Sensitivity.db_v_p_upa(-200)),
                 directivity: Directivity = Omnidirectional(),
                 rel_direction: lps_qty.RelativeBearing = lps_qty.RelativeBearing.ccw_rad(0),
                 rel_position: lps_dynamic.Displacement = \
                        lps_dynamic.Displacement(lps_qty.Distance.m(0), lps_qty.Distance.m(0))
                 ) -> None:
        self.sensitivity = sensitivity
        self.directivity = directivity
        self.rel_direction = rel_direction
        super().__init__(rel_position=rel_position)

    def transduce(self,
              input_data: np.array,
              noise_source: typing.Union[lps_dynamic.Element, lps_dynamic.RelativeElement],
              fs: lps_qty.Frequency) -> np.array:
        """ Convert input data in micro Pascal (µPa) to signal in Volts (V)
                using the sensor's sensitivity and directivity.
        """
        if self.ref_element is None:
            raise UnboundLocalError("Applying an acoustic sensor without an reference element")

        if noise_source is None:
            return self.sensitivity.convert(input_data=input_data, fs=fs)

        return self.sensitivity.convert(input_data=input_data, fs=fs) * \
                self._get_directivity(len(input_data), noise_source=noise_source)

    def _get_directivity(self,
                n_samples: int,
                noise_source: typing.Union[lps_dynamic.Element, lps_dynamic.RelativeElement]) \
                    -> np.array:
        n_steps = self.ref_element.get_n_steps()
        directivity_gain = np.zeros(n_steps)

        for step in range(n_steps):
            current_state = self[step]
            wavefront_dir = (noise_source[step].position - current_state.position).get_azimuth()
            sensor_dir = current_state.velocity.get_azimuth() + self.rel_direction
            directivity_gain[step] = self.directivity.get_gain(wavefront_dir - sensor_dir)

        directivity_gain = scipy.resample_poly(directivity_gain, n_samples, len(directivity_gain))
        return directivity_gain

    def plot_response(self, filename: str, fs: lps_qty.Frequency):
        """Plot and save the sensor's gain pattern (directivity and sensitivity)."""
        angles_deg = np.linspace(-180, 180, 360)
        angles_rad = np.deg2rad(angles_deg)

        directivity_db = np.array([
            self.directivity.get_gain(lps_qty.RelativeBearing.ccw_rad(a)) for a in angles_rad
        ])

        duration = lps_qty.Time.s(1)
        n_samples = int(duration * fs)
        input_white_noise = np.random.normal(0, 1, n_samples)

        output = self.sensitivity.convert(input_white_noise, fs.get_hz())

        fft_in = np.fft.rfft(input_white_noise)
        fft_out = np.fft.rfft(output)
        freqs = np.fft.rfftfreq(n_samples, (1/fs).get_s())

        sensitivity_db = 20 * np.log10(np.abs(fft_out)/np.abs(fft_in))

        ymin = np.min(sensitivity_db)
        ymax = np.max(sensitivity_db)

        if ymax - ymin < 10:
            center = (ymax + ymin) / 2
            ymin = center - 5
            ymax = center + 5

        tick_step = 10
        yticks = np.arange(math.floor(ymin / tick_step) * tick_step,
                        math.ceil(ymax / tick_step) * tick_step + tick_step,
                        tick_step)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1, polar=True)
        plt.plot(angles_rad, directivity_db, label='Directivity (dB)')
        plt.title("Directivity Pattern")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(freqs, sensitivity_db, label='Total Sensitivity (dB re V/µPa)')
        # plt.title("Sensitivity Frequency Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain (dB re V/µPa)")
        plt.ylim([ymin, ymax])
        plt.yticks(yticks)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


class ADConverter():
    """ Class to represent an Analog to Digital Converter"""

    def __init__(self,
                 input_limits: typing.Tuple[float, float] = None,
                 resolution: float = 16):
        self.input_limits = input_limits if input_limits is not None else [-5, 5]
        self.resolution = resolution

    def apply(self, input_data: np.array) -> np.array:
        """ Convert an array of float to an array of int as an ADC. """
        v_min, v_max = self.input_limits

        input_data = np.clip(input_data, v_min, v_max)


        max_output_value = 2**(self.resolution - 1) - 1
        min_output_value = -2**(self.resolution - 1)

        voltage_range = v_max - v_min

        scaled_data = min_output_value + \
                    ((input_data - v_min) / voltage_range) * (max_output_value - min_output_value)


        if self.resolution <= 8:
            return scaled_data.astype(np.int8)
        if self.resolution <= 16:
            return scaled_data.astype(np.int16)
        if self.resolution <= 32:
            return scaled_data.astype(np.int32)
        if self.resolution <= 64:
            return scaled_data.astype(np.int64)

        raise NotImplementedError(f"ADConverter for {self.resolution} bits")

class SignalConditioning():
    """ Class to represent the signal conditioning of an acoustic sensor. """

    @abc.abstractmethod
    def convert(self, input_data: np.array, fs: lps_qty.Frequency) -> np.array:
        """ convert an input signal in µPa para um sinal em V. """

class IdealAmplifier(SignalConditioning):
    """ Class to represent the ideal amplifier as a signal conditioning. """

    def __init__(self, gain_db: float = 0):
        super().__init__()
        self.gain_db = gain_db

    @overrides.overrides
    def convert(self, input_data: np.array, fs: lps_qty.Frequency) -> np.array:
        """ convert an input signal in µPa para um sinal em V. """
        return input_data * 10**(self.gain_db / 20)


class Sonar(lps_dynamic.Element):
    """ Class to represent a Sonar (with multiple acoustic sensors) in the scenario """

    def __init__(self,
                 sensors: typing.List[AcousticSensor],
                 signal_conditioner: SignalConditioning = IdealAmplifier(40),
                 adc: ADConverter = ADConverter(),
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> None:
        super().__init__(initial_state=initial_state)
        self.sensors = sensors
        self.adc = adc
        self.signal_conditioner = signal_conditioner

        for sensor in sensors:
            sensor.set_base_element(self)

    @staticmethod
    def hydrophone(
                 sensitivity: lps_qty.Sensitivity,
                 adc: ADConverter = ADConverter(),
                 signal_conditioner: SignalConditioning = IdealAmplifier(40),
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> 'Sonar':
        """ Class constructor for construct a Sonar with only one sensor """
        return Sonar(sensors = [AcousticSensor(sensitivity=FlatBand(sensitivity))],
                     adc = adc,
                     signal_conditioner = signal_conditioner,
                     initial_state=initial_state)

    @staticmethod
    def planar(n_staves: int,
               spacing: lps_qty.Distance,
               sensitivity: lps_qty.Sensitivity,
               adc: ADConverter = ADConverter(),
               signal_conditioner: SignalConditioning = IdealAmplifier(40),
               initial_state: lps_dynamic.State = lps_dynamic.State()) -> 'Sonar':
        """ Class constructor for construct a Sonar with only one sensor """
        sensors = []
        start_position = -((n_staves - 1) / 2) * spacing

        for stave_i in range(n_staves):
            sensors.append(AcousticSensor(sensitivity=FlatBand(sensitivity),
                                          rel_position=lps_dynamic.Displacement(
                                              start_position + stave_i * spacing,
                                              lps_qty.Distance.m(0)
                                          )))

        return Sonar(sensors = sensors,
                     adc = adc,
                     signal_conditioner = signal_conditioner,
                     initial_state=initial_state)

    @staticmethod
    def cylindrical(n_staves: int,
                    radius: lps_qty.Distance,
                    sensitivity: lps_qty.Sensitivity,
                    adc: ADConverter = ADConverter(),
                    signal_conditioner: SignalConditioning = IdealAmplifier(40),
                    initial_state: lps_dynamic.State = lps_dynamic.State()) -> 'Sonar':
        """ Class constructor for construct a Sonar with only one sensor """
        sensors = []
        angle_increment = 2 * math.pi / n_staves

        for stave_i in range(n_staves):
            angle = stave_i * angle_increment
            x_position = radius * math.cos(angle)
            y_position = radius * math.sin(angle)
            sensors.append(AcousticSensor(sensitivity=FlatBand(sensitivity),
                                    directivity=Shaded(),
                                    rel_position=lps_dynamic.Displacement(x_position,y_position),
                                    rel_direction = lps_qty.RelativeBearing.ccw_rad(angle)))

        return Sonar(sensors=sensors,
                     adc = adc,
                     signal_conditioner = signal_conditioner,
                     initial_state=initial_state)

    def get_data(self,
                 noise_compiler: lps_noise.NoiseCompiler,
                 channel: lps_channel.Channel,
                 environment: lps_env.Environment) -> np.array:
        """
        Computes the received signals for each sensor using multithreading.
        Each sensor’s received signal is computed taking into account the noise sources,
        propagation channel, and environmental conditions.

        Args:
            noise_compiler (lps_noise.NoiseCompiler): Compiled noise object containing the signal,
                source positions, and depths.
            channel (lps_channel.Channel): Acoustic propagation model that simulates how signals
                travel through the channel.
            environment (lps_env.Environment): Environmental context noise.

        Returns:
            np.array: (n_samples, n_sensors) sonar data
        """

        sonar_signals = [None] * len(self.sensors)
        with future_lib.ThreadPoolExecutor(max_workers=len(self.sensors)) as executor:
            futures = {
                executor.submit(
                    self._calculate_sensor_signal,
                    sensor, noise_compiler, channel, environment): idx
                for idx, sensor in enumerate(self.sensors)
            }

            for future in tqdm.tqdm(future_lib.as_completed(futures), total=len(futures),
                                    desc="Estimating signal for sensors", leave=False, ncols=120):
                idx = futures[future]
                sonar_signals[idx] = future.result()

        min_size = min(signal.shape[0] for signal in sonar_signals)
        signals = [signal[:min_size] for signal in sonar_signals]
        signals = np.column_stack(signals)
        return signals

    def _process_source_for_sensor(self,
                                signal: np.ndarray,
                                depth: lps_qty.Distance,
                                noise_source: typing.Union[lps_dynamic.Element,lps_dynamic.RelativeElement],
                                sensor: AcousticSensor,
                                channel: lps_channel.Channel,
                                noise_fs: lps_qty.Frequency) -> np.ndarray:
        """Processa um único ruído para um sensor"""
        rel_distance = []
        source_doppler_list = []
        sensor_doppler_list = []

        for step_id in range(self.get_n_steps()):
            rel_distance.append(
                (noise_source[step_id].position - sensor[step_id].position).get_magnitude())
            source_doppler_list.append(
                noise_source[step_id].get_relative_speed(sensor[step_id]))
            sensor_doppler_list.append(
                sensor[step_id].get_relative_speed(noise_source[step_id]))

        source_ss = channel.description.get_speed_at(depth)
        sensor_ss = channel.description.get_speed_at(channel.sensor_depth)

        doppler_noise = lps_model.apply_doppler(
            input_data=signal,
            speeds=source_doppler_list,
            sound_speed=source_ss
        )

        propag_noise = channel.propagate(
            input_data=doppler_noise,
            source_depth=depth,
            distance=rel_distance
        )

        doppler_noise = lps_model.apply_doppler(
            input_data=propag_noise,
            speeds=sensor_doppler_list,
            sound_speed=sensor_ss
        )

        return sensor.transduce(input_data=doppler_noise,
                                noise_source=noise_source,
                                fs=noise_fs)


    def _calculate_sensor_signal(self,
                                 sensor: AcousticSensor,
                                 noise_compiler: lps_noise.NoiseCompiler,
                                 channel: lps_channel.Channel,
                                 environment: lps_env.Environment,
                                 without_digitalization : bool = False):

        with future_lib.ThreadPoolExecutor(max_workers=len(noise_compiler)) as executor:
            futures = [
                executor.submit(
                    self._process_source_for_sensor,
                    signal, depth, source_list[0], sensor, channel, noise_compiler.fs
                )
                for signal, depth, source_list in noise_compiler
            ]

        noises = [f.result() for f in tqdm.tqdm(future_lib.as_completed(futures),
                                                total=len(futures),
                                                desc="Propagating noises",
                                                leave=False,
                                                ncols=120)]

        min_size = min(signal.shape[0] for signal in noises)
        signals = [signal[:min_size] for signal in noises]
        signal = np.sum(np.column_stack(signals), axis=1)

        if environment is not None:
            env_noise = environment.generate_bg_noise(min_size,
                                                    fs=noise_compiler.fs.get_hz())
            env_noise = sensor.transduce(input_data=env_noise,
                                        noise_source=None,
                                        fs=noise_compiler.fs)
            signal = signal + env_noise

        cond_signal = self.signal_conditioner.convert(signal,
                                                      fs = noise_compiler.fs)

        if without_digitalization:
            return cond_signal
        else:
            return self.adc.apply(cond_signal)
