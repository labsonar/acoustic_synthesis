"""
Background Noise Generation Module

This module provides functionality to generate background noise based on different environmental
conditions, such as rain, sea state, and shipping noise. The noise is generated based on the
frequency response data available in Chapter 7 of "Underwater Acoustics: Analysis, Design, and
Performance of SONAR" by R. P. Hodges (John Wiley and Sons, Ltd, 2010).

This module defines three Enums representing different sources of background noise:
- Rain: Enum for rain noise with various intensity levels.
- Sea: Enum for sea state noise with different states.
- Shipping: Enum for shipping noise with various intensity levels.

Each Enum provides methods to retrieve the corresponding frequency spectrum and generate noise.
"""
import os
import enum
import typing
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lps_sp.acoustical.broadband as lps
import lps_utils.quantities as lps_qty

def one_third_octave_frequencies(lower_bound = -20, upper_bound = -20) -> np.array:
    """ Get the central frequencies for a 1/3 octave, following the norm IEC 61260-1
    https://cdn.standards.iteh.ai/samples/13383/3c4ae3e762b540cc8111744cb8f0ae8e/IEC-61260-1-2014.pdf

    Args:
        lower_bound (int, optional): Number of frequencies lower than 1 kHz. Defaults to -20.
        upper_bound (int, optional): Number of frequencies higher than 1 kHz. Defaults to -20.

    Returns:
        np.array: frequencies in Hz
    """
    return np.array([1e3 * 2**(i/3) for i in range(lower_bound, upper_bound + 1)])


def turbulence_psd() -> typing.Tuple[np.array, np.array]:
    """
    Get the PSD (Power Spectral Density) of the rain noise.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Frequencies in Hz.
            - PSD estimates in dB ref 1μPa @1m/Hz.
    """
    frequencies = one_third_octave_frequencies(-30, 0)
    spectrum = np.array([107 - 30 * np.log10(f) for f in frequencies])
    return frequencies, spectrum

class Shipping(enum.Enum):
    """Enum representing Shipping level noise with various intensity levels."""
    NONE = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    LEVEL_5 = 5
    LEVEL_6 = 6
    LEVEL_7 = 7

    @staticmethod
    def __get_csv() -> str:
        return os.path.join(os.path.dirname(__file__), "data", "shipping_noise.csv")

    def __str__(self):
        if self == Shipping.NONE:
            return "without shipping noise"
        return "shipping noise " + \
            str(self.name).rsplit(".", maxsplit=1)[-1].lower().replace("_", " ")

    def get_psd(self) -> typing.Tuple[np.array, np.array]:
        """
        Get the PSD (Power Spectral Density) of the rain noise.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - PSD estimates in dB ref 1μPa @1m/Hz.
        """
        df = pd.read_csv(Shipping.__get_csv())
        frequencies = df[df.columns[0]].values
        if self != Shipping.NONE:
            spectrum = df[df.columns[self.value]].values
        else:
            spectrum = np.zeros(frequencies.size)
        return frequencies, spectrum

    @staticmethod
    def get_interpolated_psd(value: typing.Union[float, 'Shipping']) \
            -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Get the interpolated PSD for a given Shipping level noise value between 0 and 7.

        Args:
            value (float): Shipping level value between 0 and 7.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - Interpolated PSD estimates in dB ref 1μPa @1m/Hz.
        """
        if isinstance(value, Shipping):
            return value.get_psd()

        if not 0 <= value <= 7:
            raise ValueError("Shipping level noise must be between 0 and 7.")

        lower_state = Shipping(int(np.floor(value)))
        upper_state = Shipping(int(np.ceil(value)))

        frequencies, lower_psd = lower_state.get_psd()

        if lower_state == upper_state:
            return frequencies, lower_psd

        _, upper_psd = upper_state.get_psd()

        weight = value - int(np.floor(value))
        interpolated_psd = lower_psd * (1 - weight) + upper_psd * weight

        return frequencies, interpolated_psd

class Rain(enum.Enum):
    """Enum representing rain noise with various intensity levels."""

    NONE = 0
    LIGHT = 1 #(1 mm/h)
    MODERATE = 2 #(5 mm/h)
    HEAVY = 3 #(10 mm/h)
    VERY_HEAVY = 4 #(100 mm/h)

    @staticmethod
    def __get_csv() -> str:
        return os.path.join(os.path.dirname(__file__), "data", "rain.csv")

    def __str__(self):
        if self == Rain.NONE:
            return "without rain"
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower().replace("_", " ")

    def to_mm_p_h(self) -> float:
        """ Get rain intensity in mm/h. """
        value_dict = {
            Rain.NONE: 0,
            Rain.LIGHT: 1,
            Rain.MODERATE: 5,
            Rain.HEAVY: 10,
            Rain.VERY_HEAVY: 100
        }
        return value_dict[self]

    def get_psd(self) -> typing.Tuple[np.array, np.array]:
        """
        Get the PSD (Power Spectral Density) of the rain noise.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - PSD estimates in dB ref 1μPa @1m/Hz.
        """
        df = pd.read_csv(Rain.__get_csv())
        frequencies = df[df.columns[0]].values
        if self != Rain.NONE:
            psd = df[df.columns[self.value]].values
        else:
            psd = np.zeros(frequencies.size)
        return frequencies, psd

    @staticmethod
    def get_interpolated_psd(value: typing.Union[float, 'Rain']) \
            -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Get the interpolated PSD for a given Rain level noise value between 0 and 4.

        Args:
            value (float): Rain level value between 0 and 4.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - Interpolated PSD estimates in dB ref 1μPa @1m/Hz.
        """
        if isinstance(value, Rain):
            return value.get_psd()

        if not 0 <= value <= 4:
            raise ValueError("Rain level noise must be between 0 and 4.")

        lower_state = Rain(int(np.floor(value)))
        upper_state = Rain(int(np.ceil(value)))

        frequencies, lower_psd = lower_state.get_psd()

        if lower_state == upper_state:
            return frequencies, lower_psd

        _, upper_psd = upper_state.get_psd()

        weight = value - int(np.floor(value))
        interpolated_psd = lower_psd * (1 - weight) + upper_psd * weight

        return frequencies, interpolated_psd

class Sea(enum.Enum):
    """Enum representing Sea state noise with various intensity levels."""
    STATE_0 = 0
    STATE_1 = 1
    STATE_2 = 2
    STATE_3 = 3
    STATE_4 = 4
    STATE_5 = 5
    STATE_6 = 6

    @staticmethod
    def __get_csv() -> str:
        return os.path.join(os.path.dirname(__file__), "data", "sea_state.csv")

    def __str__(self):
        return f"sea state {self.value}"

    def get_psd(self) -> typing.Tuple[np.array, np.array]:
        """
        Get the PSD (Power Spectral Density) of the rain noise.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - PSD estimates in dB ref 1μPa @1m/Hz.
        """
        df = pd.read_csv(Sea.__get_csv())
        frequencies = df[df.columns[0]].values
        psd = df[df.columns[self.value + 1]].values
        return frequencies, psd

    @staticmethod
    def get_interpolated_psd(value: typing.Union[float, 'Sea']) \
            -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Get the interpolated PSD for a given sea state value between 0 and 6.

        Args:
            value (float): Sea state value between 0 and 6.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - Interpolated PSD estimates in dB ref 1μPa @1m/Hz.
        """
        if isinstance(value, Sea):
            return value.get_psd()

        if not 0 <= value <= 6:
            raise ValueError("Sea state must be between 0 and 6.")

        lower_state = Sea(int(np.floor(value)))
        upper_state = Sea(int(np.ceil(value)))

        frequencies, lower_psd = lower_state.get_psd()

        if lower_state == upper_state:
            return frequencies, lower_psd

        _, upper_psd = upper_state.get_psd()

        weight = value - int(np.floor(value))
        interpolated_psd = lower_psd * (1 - weight) + upper_psd * weight

        return frequencies, interpolated_psd

    def get_wind_speed(self) -> lps_qty.Speed:
        """ Return the mean wind speed of the air. """
        ret_dict = {
            Sea.STATE_0: 0,
            Sea.STATE_1: 2.45,
            Sea.STATE_2: 4.4,
            Sea.STATE_3: 6.7,
            Sea.STATE_4: 9.35,
            Sea.STATE_5: 12.3,
            Sea.STATE_6: 17.3,
        }
        return lps_qty.Speed.m_s(ret_dict[self])

    def get_rms_roughness(self) -> lps_qty.Distance:
        """ Return the rms roughness of the air-sea interface. """

        """BYE, “On the Variability of the Charnock ...” doi: 10.1007/s10236-014-0735-4"""
        friction_speed = self.get_wind_speed() * 0.045 - lps_qty.Speed.m_s(0.07)
        if friction_speed.get_m_s() < 0:
            return lps_qty.Distance.m(0)


        """WU “A Review of Surface Swell Waves ...” doi: 10.1016/j.ocemod.2024."""
        charnock_constant = 0.0185
        gravitational_acceleration = lps_qty.Acceleration.m_s2(9.81)

        return charnock_constant * gravitational_acceleration** 2 / gravitational_acceleration

class Environment():
    """Class to represent an acoustical environment background."""

    def __init__(self,
                 rain_value: typing.Union[float, Rain],
                 sea_value: typing.Union[float, Sea],
                 shipping_value: typing.Union[float, Shipping],
                 seed: int = None) -> None:
        """
        Args:
            rain_value (typing.Union[float, Rain]): Rain level value between 0 and 4.
            sea_value (typing.Union[float, Sea]): Sea state value between 0 and 6.
            shipping_value (typing.Union[float, Shipping]): Shipping level value between 0 and 7.
        """
        self.rain_value = rain_value
        self.sea_value = sea_value
        self.shipping_value = shipping_value
        self.seed = seed if seed is not None else id(self)
        self.rng = np.random.default_rng(seed = self.seed)

    def _format_value(value) -> str:
        """Helper to format float or Enum values."""
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    def __str__(self) -> str:
        return (f'Rain[{Environment._format_value(self.rain_value)}], '
                f'Sea[{Environment._format_value(self.sea_value)}], '
                f'Shipping[{Environment._format_value(self.shipping_value)}]')

    @classmethod
    def random(cls, seed: int = None) -> 'Environment':
        """
        Generate a sorted Background.

        Returns:
            Background: sorted Background.
        """
        rng = random.Random(seed)
        return Environment(
            rain_value = rng.uniform(Rain.NONE.value, Rain.VERY_HEAVY.value),
            sea_value = rng.uniform(Sea.STATE_0.value, Sea.STATE_6.value),
            shipping_value = rng.uniform(Shipping.NONE.value, Shipping.LEVEL_7.value),
            seed = seed
        )

    def to_psd(self) -> typing.Tuple[np.array, np.array]:
        """
        Calculate the background PSD.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - Combined PSD estimates in dB ref 1μPa @1m/Hz.
        """
        frequencies0, spectrum0 = turbulence_psd()
        frequencies1, spectrum1 = Rain.get_interpolated_psd(self.rain_value)
        frequencies2, spectrum2 = Sea.get_interpolated_psd(self.sea_value)
        frequencies3, spectrum3 = Shipping.get_interpolated_psd(self.shipping_value)

        all_frequencies = np.unique(np.concatenate([frequencies0,
                                                    frequencies1,
                                                    frequencies2,
                                                    frequencies3]))

        interpolated_spectrum0 = np.interp(all_frequencies, frequencies0, spectrum0,
                                                left=0, right=0)
        interpolated_spectrum1 = np.interp(all_frequencies, frequencies1, spectrum1,
                                                left=0, right=0)
        interpolated_spectrum2 = np.interp(all_frequencies, frequencies2, spectrum2,
                                                left=0, right=0)
        interpolated_spectrum3 = np.interp(all_frequencies, frequencies3, spectrum3,
                                                left=0, right=0)

        linear0 = 10**(interpolated_spectrum0 / 20)
        linear1 = 10**(interpolated_spectrum1 / 20)
        linear2 = 10**(interpolated_spectrum2 / 20)
        linear3 = 10**(interpolated_spectrum3 / 20)
        interpolated_spectrum = 20 * np.log10(linear0 + linear1 + linear2 + linear3)

        return all_frequencies, interpolated_spectrum

    def generate_bg_noise(self, n_samples: int = 1024, fs: float = 48000) -> np.array:
        """
        Calculate a block sample of noise for this background.

        Args:
            n_samples (int, optional): Number of samples. Defaults to 1024.
            fs (float, optional): Sample frequency. Defaults to 48 kHz.

        Returns:
            np.array: Generated broadband noise in μPa.
        """
        freq_turb, psd_turb = turbulence_psd()
        freq_rain, psd_rain = Rain.get_interpolated_psd(self.rain_value)
        freq_sea, psd_sea = Sea.get_interpolated_psd(self.sea_value)
        freq_shipping, psd_shipping = Shipping.get_interpolated_psd(self.shipping_value)

        turb_noise, _ = lps.generate(frequencies = freq_turb, psd_db = psd_turb,
                                    n_samples=n_samples, fs = fs, seed = self.rng)
        rain_noise, _ = lps.generate(frequencies = freq_rain, psd_db = psd_rain,
                                    n_samples=n_samples, fs = fs, seed = self.rng)
        sea_noise, _ = lps.generate(frequencies = freq_sea, psd_db = psd_sea,
                                    n_samples=n_samples, fs = fs, seed = self.rng)
        shipping_noise, _ = lps.generate(frequencies = freq_shipping, psd_db = psd_shipping,
                                    n_samples=n_samples, fs = fs, seed = self.rng)

        return turb_noise + rain_noise + sea_noise + shipping_noise

    def save_plot(self, filename: str) -> None:
        """ Save the expected PSD of this Enviroment. """
        f, p = self.to_psd()

        plt.figure(figsize=(10, 6))
        plt.plot(f, p)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel("Power Spectral Density (dB ref 1 µPa / √Hz)")
        plt.semilogx()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
