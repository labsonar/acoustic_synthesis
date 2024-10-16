import typing

import numpy as np

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic

class ADConverter():
    """ Class to represent an Analog to Digital Converter"""

    def __init__(self,
                 input_limits: typing.Tuple[float, float] = [-5, 5],
                 resolution: float = 16):
        self.input_limits = input_limits
        self.resolution = resolution

    def apply(self, input_data: np.array) -> np.array:
        """ Convert an array of float to an array of int as an ADC. """
        v_min, v_max = self.input_limits

        input_data = np.clip(input_data, v_min, v_max)

        print("clipped: ", np.min(input_data), " -> ", np.max(input_data))

        max_output_value = 2**(self.resolution - 1) - 1
        min_output_value = -2**(self.resolution - 1)

        voltage_range = v_max - v_min

        scaled_data = min_output_value + \
                    ((input_data - v_min) / voltage_range) * (max_output_value - min_output_value)

        print("digital: ", np.min(scaled_data), " -> ", np.max(scaled_data))

        if self.resolution <= 8:
            return scaled_data.astype(np.int8)
        if self.resolution <= 16:
            return scaled_data.astype(np.int16)
        if self.resolution <= 32:
            return scaled_data.astype(np.int32)
        if self.resolution <= 64:
            return scaled_data.astype(np.int64)

        raise NotImplementedError(f"ADConverter for {self.resolution} bits")

class AcousticSensor():
    """ Class to represent an AcousticSensor in the scenario """

    def __init__(self,
                 sensitivity: lps_qty.Sensitivity,
                 gain_db: float = 0,
                 adc: ADConverter = ADConverter(),
                 rel_position: lps_dynamic.Displacement = \
                        lps_dynamic.Displacement(lps_qty.Distance.m(0), lps_qty.Distance.m(0))
                 ) -> None:
        self.sensitivity = sensitivity
        self.gain_db = gain_db
        self.adc = adc
        self.rel_position = rel_position

    def apply(self, input_data: np.array) -> np.array:
        """ Convert input data in micropascal (µPa) to a digital signal using the sensor's
        sensitivity, gain and ADC conversion.
        """
        print("input_data: ", np.min(input_data), " -> ", np.max(input_data))

        data = input_data * 10**((self.sensitivity.get_db_v_p_upa() + self.gain_db) / 20)

        print("pos gain: ", np.min(data), " -> ", np.max(data))
        return self.adc.apply(data)

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
