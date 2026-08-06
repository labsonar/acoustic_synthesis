"""Make a new database
"""
import os
import argparse
import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import lps_utils.quantities as lps_qty
import lps_utils.quantities as lps_qty
import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.acoustical.broadband as lps_bb
import lps_ml.datasets as ml_db
import lps_ml.visualization.tsne as ml_vis
import lps_ml.datasets as ml_db
import lps_ml.core.cv as ml_cv
import lps_ml.audio_processors as ml_procs
import lps_ml.visualization.separability as ml_sep

import memory_profiler


def feature_spectrogram(x, fs):
    x = x.reshape(-1)
    power, _, _ = lps_analysis.SpectralAnalysis.SPECTROGRAM.apply(
        x, fs, lps_analysis.Parameters()
    )
    return np.mean(power, axis=1)


def feature_lofar(x, fs):
    x = x.reshape(-1)
    power, _, _ = lps_analysis.SpectralAnalysis.LOFAR.apply(
        x, fs, lps_analysis.Parameters()
    )
    return np.mean(power, axis=1)


def feature_melgram(x, fs):
    x = x.reshape(-1)
    power, _, _ = lps_analysis.SpectralAnalysis.MELGRAM.apply(
        x, fs, lps_analysis.Parameters()
    )
    return np.mean(power, axis=1)


def feature_psd(x, fs):
    _, p = lps_bb.psd(x, fs)
    return p


def feature_demon(x, fs):
    x = x.reshape(-1)

    intensity, _, _ = lps_bb.demon(
        x,
        fs,
        n_fft=1024,
        max_freq=lps_qty.Frequency.hz(100),
        overlap_ratio=0.5
    )

    return np.mean(intensity, axis=0)

def feature_vae(x, fs, encoder):

    _, z = encoder.process(
        lps_qty.Frequency.hz(fs),
        x
    )

    # opcional: reduzir dimensão (igual você faz nos outros)
    if z.ndim > 1:
        z = z.reshape(-1)

    return z

def compute_features(loader, fs, extractor, dm):

    all_data = []
    all_targets = []

    for x, target in tqdm.tqdm(loader, desc="Loading fragments"):

        x = x.numpy()
        target = target.numpy()

        for sample in x:

            feat = extractor(sample, fs)

            all_data.append(feat)

        all_targets.extend(target)

    data = np.vstack(all_data)

    return data, all_targets

def combine_labels(labels_class, labels_name):

    pairs = list(zip(labels_class, labels_name))

    combined_str = np.array([f"{c}_{n}" for c, n in pairs])

    return combined_str


def compute_separability_table(data, labels):

    unique_labels = np.unique(labels)
    metrics = [m.get() for m in ml_sep.Separability]

    results = {str(m): [] for m in metrics}

    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):

            la = unique_labels[i]
            lb = unique_labels[j]

            xa = data[labels == la]
            xb = data[labels == lb]

            for metric in metrics:
                val = metric.apply(xa, xb)
                results[str(metric)].append(val)

    return {
        k: float(np.mean(v)) for k, v in results.items()
    }


def print_table(title, table):

    print(f"\n=== {title} ===")

    metrics_names = list(next(iter(table.values())).keys())

    header = "Metric".ljust(25)
    for feat in table.keys():
        header += feat.ljust(20)
    print(header)

    for metric in metrics_names:
        row = metric.ljust(25)
        for feat in table.keys():
            val = table[feat][metric]
            row += f"{val:.4f}".ljust(20)
        print(row)

def save_heatmap(df, title, filename):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".3f", linewidths=0.5, cmap="viridis")
    plt.title(title)
    plt.ylabel("Feature")
    plt.xlabel("Metric")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def build_metric_tables(sep_tables):

    metric_tables = {}

    domains = ["class"]

    sample_feature = next(iter(sep_tables["class"].values()))
    metrics = sample_feature.keys()

    for metric in metrics:

        rows = []

        for domain in domains:

            row = {}

            for feature in sep_tables[domain]:

                val = sep_tables[domain][feature][metric]
                row[feature] = val

            rows.append(row)

        df = pd.DataFrame(rows, index=domains)

        metric_tables[metric] = df

    return metric_tables

def _main():
    parser = argparse.ArgumentParser(
        description="Synthetic database generator for underwater acoustic scenarios."
    )
    parser.add_argument("--model", type=str, default="/data/models/v0_4M6.ts")
    parser.add_argument(
        "--output-dir",
        default="/data/4classes/visualization",
        help="Directory to save results (default: /data/4classes/visualization)",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    fs = lps_qty.Frequency.khz(16)
    n_samples = int(2**17)
    overlap = int(2**16)

    dm = ml_db.FourClasses(
            file_processor=ml_procs.SampleProcessor(
                    n_samples=n_samples,
                    overlap=overlap,
                    pipelines=[
                        ml_procs.ToFloatConverter(),
                    ]
                ),
            cv = ml_cv.FiveByTwo(),
            batch_size=16,
            num_workers=0
            )

    dm.setup()

    loader = dm.all_dataloader(shuffle=False)
    loader.num_workers = 0

    # vae_encoder = ml_procs.VAEEncoder(
    #     args.model,
    #     device="cpu"  # evitar problemas de device
    # )

    features = {
        "spectrogram": feature_spectrogram,
        "lofar": feature_lofar,
        "melgram": feature_melgram,
        "psd": feature_psd,
        "demon": feature_demon,
        # "vae": lambda x, fs: feature_vae(x, fs, vae_encoder),
    }

    sep_tables = {
        "class": {},
    }

    for name, extractor in features.items():

        print(f"\nComputing features for {name}")

        data, labels = compute_features(loader, fs.get_hz(), extractor, dm)

        print(f"\tData: {data.shape}")

        sep_tables["class"][name] = compute_separability_table(data, labels)

        ml_vis.export_tsne(
            data=data,
            labels=labels,
            filename=os.path.join(output_dir, f"tsne_{name}_ship_class.png")
        )

    metric_tables = build_metric_tables(sep_tables)

    for metric, df in metric_tables.items():

        # salvar CSV
        csv_path = os.path.join(output_dir, f"separability_{metric}.csv")
        df.to_csv(csv_path)

        print(f"==== {metric} ===")
        print(df)

        # salvar heatmap
        heatmap_path = os.path.join(output_dir, f"separability_{metric}.png")

        save_heatmap(
            df,
            title=metric,
            filename=heatmap_path
        )

@memory_profiler.profile
def _run():
    _main()

if __name__ == "__main__":
    _run()
