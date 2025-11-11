import os

import lps_synthesis.dataset as syn_dataset

def _main():
    out_dir = "./result"
    os.makedirs(out_dir, exist_ok=True)

    toy_dataset = syn_dataset.ToyCatalog()
    toy_dataset_df = toy_dataset.to_df()
    toy_dataset.export(os.path.join(out_dir, "toy_dataset"))

    olocum_dataset = syn_dataset.OlocumCatalog()
    olocum_dataset_df = olocum_dataset.to_df()
    olocum_dataset.export(os.path.join(out_dir, "olocum_dataset"))

    count = toy_dataset_df.groupby(['Scenario ID']).size().reset_index(name="Qty")
    count = count.sort_values("Qty", ascending=False)
    print()
    print("############")
    print(count)

    count = toy_dataset_df.groupby(['Dynamic']).size().reset_index(name="Qty")
    print()
    print("############")
    print(count)

    count = toy_dataset_df.groupby(['Ship ID']).size().reset_index(name="Qty")
    count = count.sort_values("Qty", ascending=False)
    print()
    print("############")
    print(count)


    count = olocum_dataset_df.groupby(['Scenario ID']).size().reset_index(name="Qty")
    count = count.sort_values("Qty", ascending=False)
    print()
    print("############")
    print(count)

    count = olocum_dataset_df.groupby(['Dynamic']).size().reset_index(name="Qty")
    print()
    print("############")
    print(count)

    count = olocum_dataset_df.groupby(['Ship ID']).size().reset_index(name="Qty")
    count = count.sort_values("Qty", ascending=False)
    print()
    print("############")
    print(count)

if __name__ == "__main__":
    _main()
