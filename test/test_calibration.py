#!/usr/bin/env python3
import argparse
import os
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt


def db(x):
    return 10 * np.log10(x)


def rms(x):
    return np.sqrt(np.mean(x**2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wavfile", help="Input WAV file (int16)")
    parser.add_argument("--sens", type=float, default=-200.0,
                        help="Hydrophone sensitivity (dB re 1V/uPa)")
    parser.add_argument("--gain", type=float, default=0.0,
                        help="Preamp gain (dB)")
    parser.add_argument("--vadc", type=float, default=2.0,
                        help="ADC zero-to-peak voltage (Volts)")
    args = parser.parse_args()

    # ===============================
    # Leitura do WAV
    # ===============================
    fs, data = wav.read(args.wavfile)

    if data.dtype != np.int16:
        raise ValueError("Input WAV must be int16")

    data = data.astype(np.float64)

    duration = len(data) / fs

    print("\nCONFIGURAÇÕES GERAIS")
    print("----------------------------")
    print(f"Sensibilidade (dB re 1V/uPa): {args.sens}")
    print(f"Ganho pre (dB): {args.gain}")
    print(f"Fundo de escala ADC (V peak): {args.vadc}")
    print(f"Tipo entrada: int16")
    print(f"Fs: {fs} Hz")
    print(f"Duração: {duration:.2f} s")

    # ===============================
    # Avaliação domínio digital
    # ===============================
    print("\nAVALIAÇÃO DO DADO DE ENTRADA (Digital)")
    print("---------------------------------------")
    print(f"Média: {np.mean(data):.4f}")
    print(f"Variância: {np.var(data):.4f}")
    print(f"RMS: {rms(data):.4f}")

    # ===============================
    # Conversão para Volts
    # ===============================
    max_int = 2**15 - 1
    volts = (data / max_int) * args.vadc

    print("\nCONVERSÃO PARA VOLTS")
    print("----------------------------")
    print(f"Média: {np.mean(volts):.6f} V")
    print(f"Variância: {np.var(volts):.6e} V²")
    print(f"RMS: {rms(volts):.6f} V")
    print(f"RMS: {20*np.log10(rms(volts)):.2f} dBV")

    # ===============================
    # Conversão para pressão (µPa)
    # ===============================

    # sensibilidade: dB re 1V/uPa
    # converter para linear V/uPa
    sens_linear = 10**(args.sens / 20)

    gain_linear = 10**(args.gain / 20)

    # remover ganho do pré
    volts_input = volts / gain_linear

    # converter para µPa
    pressure_uPa = volts_input / sens_linear

    print("\nCONVERSÃO PARA PRESSÃO (µPa)")
    print("---------------------------------------")
    print(f"Média: {np.mean(pressure_uPa):.4f} µPa")
    print(f"Variância: {np.var(pressure_uPa):.4e} µPa²")
    print(f"RMS: {rms(pressure_uPa):.4f} µPa")
    print(f"SPL (RMS): {20*np.log10(rms(pressure_uPa)):.2f} dB re 1µPa @1m")

    # ===============================
    # SPL ao longo do tempo
    # ===============================
    window = int(fs * 1)
    hop = int(window * 0.5)
    spl_time = []
    times = []

    for i in range(0, len(pressure_uPa) - window, hop):
        seg = pressure_uPa[i:i+window]
        spl = 20 * np.log10(rms(seg))
        spl_time.append(spl)
        times.append(i / fs)

    print("spl_time: ", spl_time)

    plt.figure()
    plt.plot(times, spl_time)
    plt.xlabel("Time (s)")
    plt.ylabel("SPL (dB re 1µPa @1m)")
    plt.title("SPL vs Time")
    spl_path = args.wavfile.replace(".wav", "_spl.png")
    plt.savefig(spl_path, dpi=300)
    plt.close()

    # ===============================
    # PSD
    # ===============================
    f, Pxx = signal.welch(pressure_uPa,
                          fs=fs,
                          nperseg=4096,
                          scaling='density')

    Pxx_dB = 10 * np.log10(Pxx)

    plt.figure()
    plt.semilogx(f, Pxx_dB)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB re 1µPa²/Hz @1m)")
    plt.title("Power Spectral Density")
    plt.grid(True)
    psd_path = args.wavfile.replace(".wav", "_psd.png")
    plt.savefig(psd_path, dpi=300)
    plt.close()

    print("\nArquivos gerados:")
    print(spl_path)
    print(psd_path)


if __name__ == "__main__":
    main()