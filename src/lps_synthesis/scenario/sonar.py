""" Sonar Module. """
import typing
import math

import numpy as np

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic

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

class AcousticSensor(lps_dynamic.RelativeElement):
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
        super().__init__(rel_position=rel_position)

    def apply(self, input_data: np.array) -> np.array:
        """ Convert input data in micropascal (µPa) to a digital signal using the sensor's
        sensitivity, gain and ADC conversion.
        """

        data = input_data * 10**((self.sensitivity.get_db_v_p_upa() + self.gain_db) / 20)

        return self.adc.apply(data)

class Sonar(lps_dynamic.Element):
    """ Class to represent a Sonar (with multiple acoustic sensors) in the scenario """

    def __init__(self,
                 sensors: typing.List[AcousticSensor],
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> None:
        super().__init__(initial_state=initial_state)
        self.sensors = sensors

        for sensor in sensors:
            sensor.set_base_element(self)

    @staticmethod
    def hidrofone(
                 sensitivity: lps_qty.Sensitivity,
                 adc: ADConverter = ADConverter(),
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> 'Sonar':
        """ Class constructor for construct a Sonar with only one sensor """
        return Sonar(sensors = [AcousticSensor(sensitivity=sensitivity, adc=adc)],
                     initial_state=initial_state)

    @staticmethod
    def planar(n_staves: int, spacing: lps_qty.Distance,
                 sensitivity: lps_qty.Sensitivity,
                 adc: ADConverter = ADConverter(),
                 initial_state: lps_dynamic.State = lps_dynamic.State()) -> 'Sonar':
        """ Class constructor for construct a Sonar with only one sensor """
        sensors = []
        start_position = -((n_staves - 1) / 2) * spacing

        for stave_i in range(n_staves):
            sensors.append(AcousticSensor(sensitivity=sensitivity,
                                          adc=adc,
                                          rel_position=lps_dynamic.Displacement(
                                              start_position + stave_i * spacing,
                                              lps_qty.Distance.m(0)
                                          )))

        return Sonar(sensors = sensors, initial_state=initial_state)

    @staticmethod
    def cylindrical(n_staves: int,
                    radius: lps_qty.Distance,
                    sensitivity: lps_qty.Sensitivity,
                    adc: ADConverter = ADConverter(),
                    initial_state: lps_dynamic.State = lps_dynamic.State()) -> 'Sonar':
        """ Class constructor for construct a Sonar with only one sensor """
        sensors = []
        angle_increment = 2 * math.pi / n_staves

        for stave_i in range(n_staves):
            angle = stave_i * angle_increment
            x_position = radius * math.cos(angle)
            y_position = radius * math.sin(angle)
            sensors.append(AcousticSensor(sensitivity=sensitivity,
                                    adc=adc,
                                    rel_position=lps_dynamic.Displacement(x_position,y_position)))

        return Sonar(sensors=sensors, initial_state=initial_state)