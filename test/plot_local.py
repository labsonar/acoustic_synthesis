import os
import lps_synthesis.database.scenario as syndb

def _main():
    os.makedirs("./result")
    syndb.Location.plot("./result/locals.png")
    df = syndb.Location.to_df()
    df.to_csv("./result/locals.csv")

if __name__ == "__main__":
    _main()
