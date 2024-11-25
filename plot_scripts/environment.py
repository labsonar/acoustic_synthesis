""" Example of use of background module
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tikzplotlib

import lps_synthesis.environment.environment as lps


def main() -> None:

    output_dir = "./plot"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))


    line_styles = [":", "--", "-", "-."]
    rain_colors = cm.Reds(np.linspace(0.5, 1, len(lps.Rain) - 1))
    sea_colors = cm.Greens(np.linspace(0.4, 0.8, len(lps.Sea)))
    shipping_colors = cm.Blues(np.linspace(0.5, 1, len(lps.Shipping) - 1))

    f, p = lps.turbulence_psd()
    plt.plot(f, p, label="Turbulence", color="black", linestyle=line_styles[0])

    for idx, rain in enumerate(lps.Rain):
        if rain == lps.Rain.NONE:
            continue
        f, p = rain.get_psd()
        plt.plot(f, p, label=f"Rain {rain}", color=rain_colors[idx-1], linestyle=line_styles[1])

    for idx, sea in enumerate(lps.Sea):
        f, p = sea.get_psd()
        plt.plot(f, p, label=f"{sea}".capitalize(), color=sea_colors[idx], linestyle=line_styles[2])

    for idx, shipping in enumerate(lps.Shipping):
        if shipping == lps.Shipping.NONE:
            continue
        f, p = shipping.get_psd()
        plt.plot(f, p, label=f"{shipping}".capitalize(), color=shipping_colors[idx-1], linestyle=line_styles[3])

    plt.xlabel('Frequency (Hz)')
    plt.ylabel("Power Spectral Density (dB ref 1 µPa / √Hz)")
    plt.semilogx()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "environment.png"))
    tikzplotlib.save(os.path.join(output_dir, "environment.tex"))
    plt.close()

if __name__ == "__main__":
    main()
