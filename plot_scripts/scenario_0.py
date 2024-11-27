""" Example of use of scenario module
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tikz
import scipy.io.wavfile as sci_wave
import scipy.signal as sci_sig

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.scenario as lps_scenario
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.environment.environment as lps_env
import lps_synthesis.propagation.channel as lps_channel

lps_channel.TEMP_DEFAULT_DIR = "./result/propagation"


def main() -> None:

    output_dir = "./plot/scenario_0"
    os.makedirs(output_dir, exist_ok=True)

    environment = lps_env.Environment(rain_value=lps_env.Rain.LIGHT,
                                    sea_value=lps_env.Sea.STATE_2,
                                    shipping_value=lps_env.Shipping.LEVEL_5)
    channel = lps_channel.PredefinedChannel.DEEPSHIP.get_channel()
    sample_frequency = lps_qty.Frequency.khz(16)

    scenario = lps_scenario.Scenario(channel = channel,
                                    environment = environment)

    sonar = lps_sonar.Sonar.hidrofone(
            sensitivity=lps_qty.Sensitivity.db_v_p_upa(-165),
            initial_state=lps_dynamic.State(
                    position = lps_dynamic.Displacement(
                            lps_qty.Distance.m(0),
                            lps_qty.Distance.m(0)))
    )

    scenario.add_sonar("main", sonar)


    ship1 = lps_scenario.Ship(
                    ship_id="Navio 1",
                    propulsion=lps_scenario.CavitationNoise(
                        ship_type=lps_scenario.ShipType.CONTAINERSHIP,
                        n_blades=4,
                        n_shafts=1,
                    ),
                    max_speed=lps_qty.Speed.kt(15),
                    draft=lps_qty.Distance.m(2),
                    initial_state=lps_dynamic.State(
                            position = lps_dynamic.Displacement(
                                    lps_qty.Distance.km(-0.1),
                                    lps_qty.Distance.km(-0.15)),
                            velocity = lps_dynamic.Velocity(
                                    lps_qty.Speed.kt(0),
                                    lps_qty.Speed.kt(5)),
                            acceleration = lps_dynamic.Acceleration(
                                    lps_qty.Acceleration.m_s2(0),
                                    lps_qty.Acceleration.m_s2(0))
                    )
            )
    ship1.add_source(lps_scenario.Sin(frequency=lps_qty.Frequency.khz(4),
                                      amp_db_p_upa=80))
    ship1.add_source(lps_scenario.Sin(frequency=lps_qty.Frequency.khz(4.05),
                                      amp_db_p_upa=80,
                                      rel_position=lps_dynamic.Displacement(lps_qty.Distance.m(50),
                                                                            lps_qty.Distance.m(0))))

    scenario.add_noise_container(ship1)

    scenario.simulate(lps_qty.Time.s(1), 90)

    ship1[-1].acceleration = lps_dynamic.Acceleration(
                                lps_qty.Acceleration.kt_h(10*60),
                                lps_qty.Acceleration.kt_h(-10*60))

    scenario.simulate(lps_qty.Time.s(1), 30)

    ship1[-1].acceleration = lps_dynamic.Acceleration(
                                lps_qty.Acceleration.m_s2(0),
                                lps_qty.Acceleration.m_s2(0))

    scenario.simulate(lps_qty.Time.s(1), 60)

    scenario.geographic_plot(os.path.join(output_dir,"geographic.png"))
    scenario.geographic_plot(os.path.join(output_dir,"geographic.tex"))
    scenario.relative_distance_plot(os.path.join(output_dir,"distance.png"))
    scenario.relative_distance_plot(os.path.join(output_dir,"distance.tex"))
    scenario.velocity_plot(os.path.join(output_dir,"velocity.png"))
    scenario.velocity_plot(os.path.join(output_dir,"velocity.tex"))
    scenario.relative_velocity_plot(os.path.join(output_dir,"relative_velocity.png"))
    scenario.relative_velocity_plot(os.path.join(output_dir,"relative_velocity.tex"))

    output_wav = os.path.join(output_dir, "scenario.wav")

    if os.path.exists(output_wav):
        _, signal = sci_wave.read(output_wav)

    else:
        start_time = time.time()
        signal = scenario.get_sonar_audio("main", fs=sample_frequency)
        end_time = time.time()
        print("Get audio time: ", end_time-start_time)

        sci_wave.write(output_wav, int(sample_frequency.get_hz()), signal)


    time_axis = np.linspace(0, len(signal) / sample_frequency.get_hz(), num=len(signal))

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, signal, label="Audio Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Signal in Time Domain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scenario_time_serie.png"))
    plt.close()


    float_signal = signal.astype(np.float32) / (2**15 -1)
    if len(float_signal.shape) != 1:
        float_signal = float_signal[:,0]
    f, t, s = sci_sig.spectrogram(float_signal, fs=sample_frequency.get_hz(), nperseg=2048)

    aux = 20 * np.log10(np.clip(s, 1e-10, None))
    aux = aux - np.max(aux)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f/1e3, aux, shading='gouraud')
    plt.ylabel('Frequência (kHz)')
    plt.xlabel('Tempo (s)')
    cbar = plt.colorbar(label="Intensidade (dB)")
    cbar.ax.yaxis.set_label_position('left')
    plt.savefig(os.path.join(output_dir, "scenario_spectrum.png"))
    tikz.save(os.path.join(output_dir, "scenario_spectrum.tex"))
    plt.close()


    fs = int(sample_frequency.get_hz())
    f, t, s = sci_sig.spectrogram(float_signal, fs=fs, nperseg=int(2*fs), noverlap=int(2*fs * 0.5))

    aux = 20 * np.log10(np.clip(s, 1e-10, None))
    aux = aux - np.max(aux)

    freq_min = 3980
    freq_max = 4070
    freq_mask = (f >= freq_min) & (f <= freq_max)

    plt.pcolormesh(t, f[freq_mask]/1e3, aux[freq_mask, :], shading='gouraud')
    plt.ylabel('Frequência (kHz)')
    plt.xlabel('Tempo (s)')
    cbar = plt.colorbar(label="Intensidade (dB)")
    cbar.ax.yaxis.set_label_position('left')
    plt.savefig(os.path.join(output_dir, "line1_spectrogram.png"))
    tikz.save(os.path.join(output_dir, "lines_spectrogram.tex"))
    plt.close()

    freq_min = 3990
    freq_max = 4010
    freq_mask = (f >= freq_min) & (f <= freq_max)

    plt.pcolormesh(t, f[freq_mask], aux[freq_mask, :], shading='gouraud')
    plt.ylabel('Frequência (kHz)')
    plt.xlabel('Tempo (s)')
    cbar = plt.colorbar(label="Intensidade (dB)")
    cbar.ax.yaxis.set_label_position('left')
    plt.savefig(os.path.join(output_dir, "line2_spectrogram.png"))
    tikz.save(os.path.join(output_dir, "line_spectrogram.tex"))
    plt.close()

    freq_target_4050_idx = np.abs(f - 4050).argmin()
    freq_target_4000_idx = np.abs(f - 4000).argmin()

    interval_1 = (t >= 0) & (t <= 90)
    interval_2 = (t >= 120) & (t <= 180)

    intensity_4050_interval_1 = s[freq_target_4050_idx, interval_1]
    intensity_4050_interval_2 = s[freq_target_4050_idx, interval_2]

    intensity_4000_interval_1 = s[freq_target_4000_idx, interval_1]
    intensity_4000_interval_2 = s[freq_target_4000_idx, interval_2]

    time_max_4050_interval_1 = t[interval_1][np.argmax(intensity_4050_interval_1)]
    time_max_4050_interval_2 = t[interval_2][np.argmax(intensity_4050_interval_2)]

    time_max_4000_interval_1 = t[interval_1][np.argmax(intensity_4000_interval_1)]
    time_max_4000_interval_2 = t[interval_2][np.argmax(intensity_4000_interval_2)]

    print(f"Máximo para 4050 Hz no intervalo 0-90s: {time_max_4050_interval_1:.3f} s")
    print(f"Máximo para 4050 Hz no intervalo 120-180s: {time_max_4050_interval_2:.3f} s")
    print(f"Máximo para 4000 Hz no intervalo 0-90s: {time_max_4000_interval_1:.3f} s")
    print(f"Máximo para 4000 Hz no intervalo 120-180s: {time_max_4000_interval_2:.3f} s")

    interv1 = lps_qty.Time.s(np.abs(time_max_4050_interval_1 - time_max_4000_interval_1))
    interv2 = lps_qty.Time.s(np.abs(time_max_4050_interval_2 - time_max_4000_interval_2))
    v = lps_qty.Speed.kt(5)

    print(f"Diff part 1: {interv1} -> {v * interv1}")
    print(f"Diff part 2: {interv2} -> {v * interv2}")
    print(t[1]-t[0])

if __name__ == "__main__":
    main()
