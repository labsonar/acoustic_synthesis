import os
import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_utils.subprocess as lps_proc
import lps_synthesis.propagation.channel_description as lps_desc
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.oases as oases
import lps_synthesis.propagation.traceo as traceo

def freq_to_time(h_f, freqs):

    n_f = len(h_f)
    N = 2 * (n_f - 1)

    window = np.hanning(n_f)
    h_f_win = h_f * window

    H = np.zeros(N, dtype=np.complex128)
    H[:n_f] = h_f_win
    H[n_f:] = np.flip(np.conj(h_f_win[1:-1]))

    H *= N / 2

    h_t = np.fft.ifft(H)

    df = freqs[1] - freqs[0]
    fs = 2 * freqs[-1]
    t = np.arange(N) / fs

    return np.real(h_t), t

def _main(sensor_depth: lps_qty.Distance, dist: lps_qty.Distance):
    output_dir = "result/traceo"
    os.makedirs(output_dir, exist_ok=True)

    max_depth = lps_qty.Distance.m(50000)
    # sensor_depth = max_depth/2
    # sensor_depth = lps_qty.Distance.m(2)
    zoom_samples = 400

    desc = lps_desc.Description()
    desc.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))
    desc.add(max_depth, lps_qty.Speed.m_s(1500))
    # desc.add(lps_qty.Distance.m(1) + max_depth, lps_layer.BottomType.BASALT)
    # desc.remove_air_sea_interface()

    h, ranges, depths, freqs = traceo.get_response(
            sample_frequency = lps_qty.Frequency.khz(16),
            description = desc,
            source_depths = oases.Sweep(sensor_depth, lps_qty.Distance.m(1), 1),
            sensor_depth = sensor_depth,
            distance = oases.Sweep(dist, lps_qty.Distance.m(900), 1),
            aux_dir = output_dir
    )

    h_oases, ranges_oases, depths_oases, freqs_oases = oases.get_response(
            sample_frequency = lps_qty.Frequency.khz(16),
            description = desc,
            source_depths = oases.Sweep(sensor_depth, lps_qty.Distance.m(1), 1),
            sensor_depth = sensor_depth,
            distance = oases.Sweep(dist, lps_qty.Distance.m(900), 1),
            aux_dir = output_dir
    )

    print("##### traceo")
    print("h: ", h.shape)
    print("h: ", h)
    print("ranges: ", ranges)
    print("depths: ", depths)

    print("##### oases")
    print("h: ", h_oases.shape)
    print("h: ", h_oases)
    print("ranges: ", ranges_oases)
    print("depths: ", depths_oases)

    freqs = np.array([int(f.get_hz()) for f in freqs])
    freqs_oases = np.array([int(f.get_hz()) for f in freqs_oases])

    for id_z, depth in enumerate(depths):
        for id_r, range in enumerate(ranges):

            h_f = h[id_r, id_z, :]   # H(f) para esse (r, d)
            h_f_oases = h_oases[id_r, id_z, :]   # H(f) para esse (r, d)

            print("[", range,",", depth, "] traceo: ", -20.0 * np.log10(np.abs(np.mean(h_f))))
            print("[", range,",", depth, "] oases: ", -20.0 * np.log10(np.abs(np.mean(h_f_oases))))

            h_t, t = freq_to_time(h_f, freqs)

            peak_idx = np.argmax(np.abs(h_t))

            half = zoom_samples // 2
            i0 = max(0, peak_idx - half)
            i1 = min(len(h_t), peak_idx + half)
            t_zoom = t[i0:i1]
            h_t_zoom = h_t[i0:i1]


            fig, axs = plt.subplots(5, 1, figsize=(9, 8), sharex=False)

            axs[0].plot(freqs, np.real(h_f))
            axs[0].plot(freqs_oases, np.real(h_f_oases))
            axs[0].set_ylabel("Re{H}")

            axs[1].plot(freqs, np.imag(h_f))
            axs[1].plot(freqs_oases, np.imag(h_f_oases))
            axs[1].set_ylabel("Im{H}")

            axs[2].plot(freqs, -20.0 * np.log10(np.abs(h_f)))
            axs[2].plot(freqs_oases, -20.0 * np.log10(np.abs(h_f_oases)))
            axs[2].set_ylabel("|H|")
            axs[2].set_xlabel("Frequência (kHz)")

            axs[3].plot(t, h_t)
            axs[3].set_ylabel("h(t)")
            axs[3].set_xlabel("Tempo (s)")

            axs[4].plot(t_zoom, h_t_zoom)
            axs[4].set_ylabel("h(t) zoom")
            axs[4].set_xlabel("Tempo (s)")

            title = (
                f"Resposta em frequência\n"
                f"Profundidade = {depth} | "
                f"Distância = {range} | "
                f"TL = {-20.0 * np.log10(np.mean(np.abs(h_f)))}"
            )
            fig.suptitle(title)

            for ax in axs:
                ax.grid(True)

            fig.tight_layout(rect=[0, 0, 1, 0.93])

            fname = (
                f"H({int(depth.get_m())}_"
                f"{int(range.get_m())}).png"
            )
            plt.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close(fig)


if __name__ == "__main__":

    for depth in [2, 20, 25000]:
        # for dist in [10, 100, 500, 1000]:
        _main(lps_qty.Distance.m(depth), lps_qty.Distance.m(100))
