import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dyn
import lps_synthesis.propagation.layers as syn_lay
import lps_synthesis.environment.acoustic_site as syn_sites
import lps_synthesis.database.scenario as syn_scenario


def _test_seabed():
    seabed = syn_sites.SeabedProspector()
    rows = []

    for loc in syn_scenario.Location:
        p = loc.get_point()

        try:
            seabed_type = seabed.get(p)[0].name
        except ValueError:
            seabed_type = "None"


        rows.append({
            "location": loc.name,
            "lat": p.latitude.get_deg(),
            "lon": p.longitude.get_deg(),
            "seabed": seabed_type
        })

    df = pd.DataFrame(rows)
    print("\n=== SEABED RESULTS ===")
    print(df)
    return df


def _test_depth():
    depthp = syn_sites.DepthProspector()
    rows = []

    for loc in syn_scenario.Location:
        p = loc.get_point()
        depth = depthp.get(p)

        rows.append({
            "location": loc.name,
            "lat": p.latitude.get_deg(),
            "lon": p.longitude.get_deg(),
            "depth_m": depth.get_m()
        })

    df = pd.DataFrame(rows)
    print("\n=== DEPTH RESULTS ===")
    print(df)
    return df


def _test_ssp(force: bool):
    ssp = syn_sites.SSPProspector()

    # Cores fixas por local
    cmap = plt.get_cmap("tab20")
    color_cycle = cmap(np.linspace(0, 1, len(syn_scenario.Location)))

    # Estilos por estação
    season_linestyles = {
        syn_sites.Season.SUMMER: "-",
        syn_sites.Season.WINTER: "--",
        syn_sites.Season.SPRING: "-.",
        syn_sites.Season.AUTUMN: ":",
    }

    # Separar locais shallow e deep
    shallow_locs = [loc for loc in syn_scenario.Location if loc.is_shallow_water()]
    deep_locs = [loc for loc in syn_scenario.Location if not loc.is_shallow_water()]

    depth_prospector = syn_sites.DepthProspector()

    def plot_group(group_locs, fig_name):
        plt.figure(figsize=(10, 12))

        for idx, loc in enumerate(group_locs):
            print("location: ", loc)
            p = loc.get_point()
            color = color_cycle[idx % len(color_cycle)]

            for season, linestyle in season_linestyles.items():
                depths, svp = ssp.get(p,
                                      season,
                                      max_depth=None if not force else depth_prospector.get(p))

                if len(depths) == 0:
                    continue

                depths = [d.get_m() for d in depths]
                svp = [s.get_m_s() for s in svp]
                plt.plot(
                    svp,
                    depths,
                    linestyle=linestyle,
                    color=color,
                    label=f"{loc.name} - {season.name}"
                )

        plt.gca().invert_yaxis()
        plt.xlabel("Sound speed (m/s)")
        plt.ylabel("Depth (m)")
        plt.title(fig_name)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(
                "./result",
                fig_name.lower().replace(" ", "_") + (".png" if not force else "_force.png")),
            dpi=200)
        plt.close()

    # Gerar figuras
    plot_group(shallow_locs, "Shallow Water SSP Profiles")
    plot_group(deep_locs, "Deep Water SSP Profiles")

    print("\nSSP plots saved: shallow and deepwater.\n")

def _test_env():
    envp = syn_sites.EnvironmentProspector()
    rows = []

    for loc in syn_scenario.Location:
        for season in syn_sites.Season:

            p = loc.get_point()
            rain = envp.get_rain(p, season)
            sea = envp.get_seastate(p, season)

            rows.append({
                "location": loc.name,
                "lat": p.latitude.get_deg(),
                "lon": p.longitude.get_deg(),
                "season": season,
                "rain": rain,
                "sea": sea,
            })

    df = pd.DataFrame(rows)
    print("\n=== DEPTH RESULTS ===")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(df)
    return df

def _main():
    _test_seabed()
    _test_depth()
    _test_ssp(False)
    _test_ssp(True)
    _test_env()

    print("\nAll tests completed.\n")


if __name__ == "__main__":
    _main()
