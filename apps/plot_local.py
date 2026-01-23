import os
import lps_synthesis.database.scenario as syndb

def _main():
    base_dir = "./result"
    os.makedirs(base_dir, exist_ok=True)
    syndb.Location.plot(os.path.join(base_dir, "locals.png"))
    df = syndb.Location.to_df()
    df.to_csv(os.path.join(base_dir, "locals.csv"), index=False)

if __name__ == "__main__":
    _main()
