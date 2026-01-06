import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pandas as pd

import lps_synthesis.environment.acoustic_site as syn_site
import lps_synthesis.scenario.dynamic as syn_dyn
import lps_synthesis.database.scenario as syn_sce

def _main():

    locations = [
        ["HEBRIDES_SEA", 56.875, -7.625, "Basalt", 28.5],
        ["GULF_OF_CADIZ", 36.875, -8.325, "Basalt", 108.6],
        ["GULF_OF_ADEN", 12.375, 51.125, "Basalt", 227.6],
        ["YUCATAN_CHANNEL", 21.875, -85.125, "Basalt", 1578.2],
        ["BALTIC_SEA", 55.125, 12.875, "Silt", 35.4],
        ["GULF_OF_GUINEA", 3.875, 7.375, "Silt", 120.4],
        ["TASMAN_SEA", -39.875, 168.125, "Silt", 850.1],
        ["BAY_OF_BENGAL", 18.875, 89.375, "Silt", 1902],
        ["ARABIAN_SEA", 23.375, 67.875, "Clay", 21],
        ["GUANABARA_BAY", -23.125, -43.125, "Clay", 56.3],
        ["SEA_OF_JAPAN", 43.875, 141.125, "Clay", 288.9],
        ["SOUTH_CHINA_SEA", 10.675, 117.675, "Clay", 1183.8],
        ["GULF_OF_BOTHNIA", 64.675, 23.875, "Gravel", 26.3],
        ["ARGENTINE_SEA", -50.875, -67.625, "Gravel", 100.1],
        ["QUEEN_CHARLOTTE_SOUND", 51.375, -129.125, "Gravel", 205.4],
        ["SCOTIA_SEA", -53.625, -38.875, "Gravel", 1332],
        ["RÍO_DE_LA_PLATA_ESTUARY", -36.325, -55.875, "Sand", 16.9],
        ["CORAL_SEA", -21.875, 151.125, "Sand", 68.2],
        ["EAST_CHINA_SEA", 25.125, 122.125, "Sand", 184.6],
        ["WESTERN_SOUTH_PACIFIC_OCEAN", -12.125, 13.125, "Sand", 1166.8],

    ]

    plt.figure(figsize=(16, 9), dpi=600)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(resolution='110m', linewidth=0.6)
    ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')

    for name, lat, lon, _, depth in locations:

        color='blue' if depth < 180 else 'red'
        offset=1.5 if depth < 180 else -4.5
        ax.plot(lon, lat, 'o',
                markersize=4, color=color, transform=ccrs.PlateCarree())
        ax.text(lon + offset, lat, name,
                fontsize=6, color=color, transform=ccrs.PlateCarree())

    plt.savefig("./result/new_locals.png", dpi=600, bbox_inches="tight")

    prospector = syn_site.AcousticSiteProspector()

    infos = []
    for name, lat, lon, _, _ in locations:
        info = prospector.get_complete_info(name=name, point=syn_dyn.Point.deg(lat, lon))
        infos.append(info)

    df = pd.DataFrame(infos)
    print(df)

    df.to_csv("./result/new_locals.csv")


    infos = []
    for location in syn_sce.Location:
        info = prospector.get_complete_info(name=str(location), point=location.get_point())
        infos.append(info)

    df = pd.DataFrame(infos)
    print(df)

    df.to_csv("./result/original_locals.csv")

if __name__ == "__main__":
    _main()
