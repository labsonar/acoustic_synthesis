import os
import numpy as np

from lps_synthesis.propagation.channel_response import TemporalResponse


def correct_temporal_responses(input_dir: str, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):

        if not filename.lower().endswith(".pkl"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            tr = TemporalResponse.load(input_path)

            if tr is None or tr.h_t_tau is None:
                print(f"[SKIP] {filename} não é TemporalResponse válido.")
                continue

            # número de amostras no tempo
            n_samples = tr.h_t_tau.shape[2]

            correction_factor = n_samples / 2.0

            print(f"[OK] Corrigindo {filename} (dividindo por {correction_factor})")

            # aplica correção
            tr.h_t_tau = tr.h_t_tau / correction_factor

            # salva corrigido
            tr.save(output_path)

        except Exception as e:
            print(f"[ERROR] Falha ao processar {filename}: {e}")

if __name__ == "__main__":

    input_directory = "./channel"
    output_directory = "./channel_corr"

    correct_temporal_responses(input_directory, output_directory)
