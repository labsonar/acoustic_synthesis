"""Simple Test for ships description in datasets
"""
import lps_synthesis.database.ship as syndb_ship

if __name__ == "__main__":

    catalog = syndb_ship.ShipCatalog()
    print(f"Loaded {len(catalog)} ships")
    for s in catalog:
        print(s)

    random_catalog = syndb_ship.ShipCatalog(n_samples=50, seed=42)
    print("\nRandomized sample:")
    for s in random_catalog:
        print(f"{s.ship_name}: {s.mcr_percent:.2f} %, {s.cruising_speed}, {s.n_blades} blades")

    df = catalog.to_df()
    df.to_csv("./result/ship_catalog.csv", index=False)

    df = random_catalog.to_df()
    df.to_csv("./result/ship_random_catalog.csv", index=False)

    count = df.groupby(['iara_ship_id']).size().reset_index(name="Qty")
    count = count.sort_values("Qty", ascending=False)
    print()
    print("############")
    print(count)
