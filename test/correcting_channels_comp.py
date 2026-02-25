import numpy as np
import pickle
import os

def compare_temporal_responses(file1: str, file2: str):

    from lps_synthesis.propagation.channel_response import TemporalResponse

    tr1 = TemporalResponse.load(file1)
    tr2 = TemporalResponse.load(file2)

    if tr1 is None or tr2 is None:
        print("Erro ao carregar um dos arquivos.")
        return

    print("==== Comparação estrutural ====")

    print("Shape h_t_tau:")
    print("  TR1:", None if tr1.h_t_tau is None else tr1.h_t_tau.shape)
    print("  TR2:", None if tr2.h_t_tau is None else tr2.h_t_tau.shape)

    print("\nSample frequency:")
    print("  TR1:", tr1.sample_frequency)
    print("  TR2:", tr2.sample_frequency)

    print("\nNúmero de depths:")
    print("  TR1:", len(tr1.depths) if tr1.depths else None)
    print("  TR2:", len(tr2.depths) if tr2.depths else None)

    print("\nNúmero de ranges:")
    print("  TR1:", len(tr1.ranges) if tr1.ranges else None)
    print("  TR2:", len(tr2.ranges) if tr2.ranges else None)

    print("\n==== Comparação numérica ====")

    if tr1.h_t_tau.shape == tr2.h_t_tau.shape:
        diff = tr1.h_t_tau - tr2.h_t_tau
        print("Máx diferença absoluta:", np.max(np.abs(diff)))
        print("Erro RMS:", np.sqrt(np.mean(diff**2)))
        print("Arrays exatamente iguais?", np.array_equal(tr1.h_t_tau, tr2.h_t_tau))
        print("Arrays quase iguais (1e-12)?", np.allclose(tr1.h_t_tau, tr2.h_t_tau, atol=1e-12))
    else:
        print("Shapes diferentes — comparação numérica não realizada.")

    print("\n==== Comparação binária do pickle ====")

    with open(file1, "rb") as f:
        bytes1 = f.read()

    with open(file2, "rb") as f:
        bytes2 = f.read()

    print("Arquivos pickle idênticos byte-a-byte?", bytes1 == bytes2)
    print("Tamanho arquivo 1:", len(bytes1))
    print("Tamanho arquivo 2:", len(bytes2))


if __name__ == "__main__":
    compare_temporal_responses("./channel/coral_sea_spring_822076307648613867_oases.pkl",
                               "./channel_/coral_sea_spring_822076307648613867_oases.pkl",)