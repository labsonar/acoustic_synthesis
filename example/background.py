import os
import numpy as np
import matplotlib.pyplot as plt

import lps_synthesis.environment.background as lps

def plot_psd(values: np.ndarray, get_psd_func, title: str) -> None:
    """
    Plot PSD for a range of values using the provided PSD function.

    Args:
        values (np.ndarray): Range of values to evaluate.
        get_psd_func (function): Function to get interpolated PSD.
        title (str): Title for the plot.
    """
    base_dir = "./results"
    os.makedirs(base_dir, exist_ok = True)

    plt.figure(figsize=(10, 6))
    for value in values:
        frequencies, psd = get_psd_func(value)
        plt.plot(frequencies, psd, label=f'{value:.1f}')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB re 1μPa^2/Hz)')
    plt.title(f'{title} Noise PSD')
    plt.semilogx()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{base_dir}/{title}.png')

def main() -> None:
    """ Generate a range of values for each enum. """

    rain_values = np.arange(1, 4.1, 0.25)
    sea_values = np.arange(1, 6.1, 0.25)
    shipping_values = np.arange(1, 7.1, 0.25)

    plot_psd(rain_values, lps.Rain.get_interpolated_psd, "Rain")
    plot_psd(sea_values, lps.Sea.get_interpolated_psd, "Sea State")
    plot_psd(shipping_values, lps.Shipping.get_interpolated_psd, "Shipping")

if __name__ == "__main__":
    main()
