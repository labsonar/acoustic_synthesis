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

import lps_sp.acoustical.broadband as lps


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
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower().replace("_", " ") + " rain"

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
    def get_interpolated_psd(value: float) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Get the interpolated PSD for a given Rain level noise value between 0 and 4.

        Args:
            value (float): Rain level value between 0 and 4.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - Interpolated PSD estimates in dB ref 1μPa @1m/Hz.
        """
        if not 0 <= value <= 7:
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
    def get_interpolated_psd(value: float) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Get the interpolated PSD for a given sea state value between 0 and 6.

        Args:
            value (float): Sea state value between 0 and 6.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - Interpolated PSD estimates in dB ref 1μPa @1m/Hz.
        """
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
    def get_interpolated_psd(value: float) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Get the interpolated PSD for a given Shipping level noise value between 0 and 7.

        Args:
            value (float): Shipping level value between 0 and 7.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - Interpolated PSD estimates in dB ref 1μPa @1m/Hz.
        """
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

class Background():
    """Class to represent an acoustical environment background."""

    def __init__(self,
                 rain_value: typing.Union[float, Rain],
                 sea_value: typing.Union[float, Sea],
                 shipping_value: typing.Union[float, Shipping]) -> None:
        """
        Args:
            rain_value (typing.Union[float, Rain]): Rain level value between 0 and 4.
            sea_value (typing.Union[float, Sea]): Sea state value between 0 and 6.
            shipping_value (typing.Union[float, Shipping]): Shipping level value between 0 and 7.
        """
        self.rain_value = rain_value.value if isinstance(rain_value, Rain) else rain_value
        self.sea_value = sea_value.value if isinstance(rain_value, Sea) else sea_value
        self.shipping_value = shipping_value.value if isinstance(rain_value, Shipping) \
                                                        else shipping_value

    def __str__(self) -> str:
        return f'Rain[{self.rain_value:.1f}], ' \
                'Sea[{self.sea_value:.1f}], ' \
                'Shipping[{self.shipping_value:.1f}]'

    @classmethod
    def random(cls, ) -> 'Background':
        """
        Generate a sorted Background.

        Returns:
            Background: sorted Background.
        """
        return Background(
            rain_value=random.uniform(Rain.NONE.value, Rain.VERY_HEAVY.value),
            sea_value=random.uniform(Sea.STATE_0.value, Sea.STATE_6.value),
            shipping_value=random.uniform(Shipping.NONE.value, Shipping.LEVEL_7.value)
        )

    def to_psd(self) -> typing.Tuple[np.array, np.array]:
        """
        Calculate the background PSD.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - Combined PSD estimates in dB ref 1μPa @1m/Hz.
        """
        frequencies1, spectrum1 = Rain.get_interpolated_psd(self.rain_value)
        frequencies2, spectrum2 = Sea.get_interpolated_psd(self.sea_value)
        frequencies3, spectrum3 = Shipping.get_interpolated_psd(self.shipping_value)

        all_frequencies = np.unique(np.concatenate([frequencies1, frequencies2, frequencies3]))

        interpolated_spectrum1 = np.interp(all_frequencies, frequencies1, spectrum1,
                                                left=0, right=0)
        interpolated_spectrum2 = np.interp(all_frequencies, frequencies2, spectrum2,
                                                left=0, right=0)
        interpolated_spectrum3 = np.interp(all_frequencies, frequencies3, spectrum3,
                                                left=0, right=0)

        linear1 = 10**(interpolated_spectrum1 / 20)
        linear2 = 10**(interpolated_spectrum2 / 20)
        linear3 = 10**(interpolated_spectrum3 / 20)
        interpolated_spectrum = 20 * np.log10(linear1 + linear2 + linear3)

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
        freq_rain, psd_rain = Rain.get_interpolated_psd(self.rain_value)
        freq_sea, psd_sea = Sea.get_interpolated_psd(self.sea_value)
        freq_shipping, psd_shipping = Shipping.get_interpolated_psd(self.shipping_value)

        rain_noise = lps.generate(frequencies = freq_rain, psd_db = psd_rain,
                                    n_samples=n_samples, fs = fs)
        sea_noise = lps.generate(frequencies = freq_sea, psd_db = psd_sea,
                                    n_samples=n_samples, fs = fs)
        shipping_noise = lps.generate(frequencies = freq_shipping, psd_db = psd_shipping,
                                    n_samples=n_samples, fs = fs)

        return rain_noise + sea_noise + shipping_noise
