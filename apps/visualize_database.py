"""Make a new database
"""
import os
import argparse
import tqdm

import numpy as np

import lps_utils.quantities as lps_qty
import lps_utils.quantities as lps_qty
import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.acoustical.broadband as lps_bb
import lps_ml.datasets as ml_db
import lps_ml.visualization.tsne as ml_vis
import lps_ml.datasets as ml_db
import lps_ml.core.cv as ml_cv
import lps_ml.audio_processors as ml_procs

import memory_profiler


def feature_spectrogram(x, fs):
    power, _, _ = lps_analysis.SpectralAnalysis.SPECTROGRAM.apply(
        x, fs, lps_analysis.Parameters()
    )
    return np.mean(power, axis=1)


def feature_lofar(x, fs):
    power, _, _ = lps_analysis.SpectralAnalysis.LOFAR.apply(
        x, fs, lps_analysis.Parameters()
    )
    return np.mean(power, axis=1)


def feature_melgram(x, fs):
    power, _, _ = lps_analysis.SpectralAnalysis.MELGRAM.apply(
        x, fs, lps_analysis.Parameters()
    )
    return np.mean(power, axis=1)


def feature_psd(x, fs):
    _, p = lps_bb.psd(x, fs)
    return p


def feature_demon(x, fs):

    intensity, _, _ = lps_bb.demon(
        x,
        fs,
        n_fft=1024,
        max_freq=lps_qty.Frequency.hz(100),
        overlap_ratio=0.5
    )

    return np.mean(intensity, axis=0)


FEATURES = {
    "spectrogram": feature_spectrogram,
    "lofar": feature_lofar,
    "melgram": feature_melgram,
    "psd": feature_psd,
    "demon": feature_demon
}


def compute_features(loader, fs, extractor, dm):

    all_data = []
    all_file_ids = []

    for x, _ in tqdm.tqdm(loader, desc="Loading fragments"):

        x = x.numpy()

        for sample in x:

            feat = extractor(sample, fs)

            all_data.append(feat)

        batch_size = x.shape[0]

        start_idx = len(all_file_ids)
        ids = dm.dataframe.iloc[start_idx:start_idx + batch_size]["file_id"].values

        all_file_ids.extend(ids)

    data = np.vstack(all_data)

    return data, all_file_ids

def combine_labels(labels_class, labels_name):

    pairs = list(zip(labels_class, labels_name))

    combined_str = np.array([f"{c}_{n}" for c, n in pairs])

    return combined_str

def _main():
    parser = argparse.ArgumentParser(
        description="Synthetic database generator for underwater acoustic scenarios."
    )

    parser.add_argument(
        "--output-dir",
        default="/data/iemanja/visualization",
        help="Directory to save results (default: /data/iemanja/visualization)",
    )

    args = parser.parse_args()
    output_dir = args.output_dir

    fs=lps_qty.Frequency.khz(16)
    duration=lps_qty.Time.s(10)
    overlap=lps_qty.Time.s(5)

    dm = ml_db.Iemanja(
            file_processor=ml_procs.TimeProcessor(
                    fs_out=fs,
                    duration=duration,
                    overlap=overlap,
                    pipelines=[ml_procs.ToFloatConverter()]
                ),
            cv = ml_cv.FiveByTwo(),
            simple_version=True,
            batch_size=16,
            num_workers=0
            )

    dm.setup()

    loader = dm.all_dataloader(shuffle=False)
    loader.num_workers = 0

    df_meta = dm.to_df()

    id_to_class = dict(zip(df_meta["ID"], df_meta["CLASS"]))
    id_to_name = dict(zip(df_meta["ID"], df_meta["NAME_(US)"]))



    for name, extractor in FEATURES.items():

        print(f"\nComputing features for {name}")

        data, file_ids = compute_features(loader, fs.get_hz(), extractor, dm)

        labels_class = np.array([id_to_class[i] for i in file_ids])
        labels_name = np.array([id_to_name[i] for i in file_ids])

        labels_combined_str = combine_labels(
            labels_class,
            labels_name
        )

        # ml_vis.export_tsne(
        #     data=data,
        #     labels=labels_class,
        #     filename=os.path.join(output_dir, f"tsne_{name}_ship_class.png")
        # )

        # ml_vis.export_tsne(
        #     data=data,
        #     labels=labels_name,
        #     filename=os.path.join(output_dir, f"tsne_{name}_channel.png")
        # )

        ml_vis.export_tsne(
            data=data,
            labels=labels_combined_str,
            filename=os.path.join(output_dir, f"tsne_{name}_combined.png")
        )



@memory_profiler.profile
def _run():
    _main()

if __name__ == "__main__":
    _run()
