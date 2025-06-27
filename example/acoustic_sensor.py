import os

import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.sonar as lps_sonar
import lps_synthesis.scenario.noise_source as lps_noise
import lps_synthesis.scenario.dynamic as lps_dynamic

def main():
    """main function of test noise_source."""
    base_dir = "./result/acoustic_sensor"
    os.makedirs(base_dir, exist_ok=True)

    fs = lps_qty.Frequency.khz(16)
    duration = lps_qty.Time.s(1)
    n_samples = int(duration * fs)

    omni_sensor = lps_sonar.AcousticSensor(
        sensitivity = lps_sonar.FlatBand(lps_qty.Sensitivity.db_v_p_upa(-200)),
        directivity = lps_sonar.Omnidirectional()
    )
    omni_sensor.plot_response(os.path.join(base_dir,"omni_sensor.png"), fs=fs)

    shaded_sensor = lps_sonar.AcousticSensor(
        sensitivity = lps_sonar.FlatBand(lps_qty.Sensitivity.db_v_p_upa(-200)),
        directivity = lps_sonar.Shaded(d=10, m=8.7)
    )
    shaded_sensor.plot_response(os.path.join(base_dir,"shaded_sensor.png"), fs=fs)


    sin_noise = lps_noise.NarrowBandNoise(
        frequency=lps_qty.Frequency.khz(2),
        amp_db_p_upa=150
    )


    R = 25
    T = duration.get_s()
    omega = 2 * np.pi / T

    pos0 = -8
    vel0 = 2
    acc0 = 0.25

    source = lps_dynamic.Element(
        initial_state=lps_dynamic.State(
            position = lps_dynamic.Displacement(
                lps_qty.Distance.m(pos0), lps_qty.Distance.m(0)),
            velocity = lps_dynamic.Velocity(
                lps_qty.Speed.m_s(0), lps_qty.Speed.m_s(vel0)),
            acceleration = lps_dynamic.Acceleration(
                lps_qty.Acceleration.m_s2(acc0), lps_qty.Acceleration.m_s2(-acc0))
        )
    )
    sin_noise.set_base_element(source)
    source.move(lps_qty.Time.s(1), 8)
    source.state_list[-1].acceleration = lps_dynamic.Acceleration(
                lps_qty.Acceleration.m_s2(-acc0), lps_qty.Acceleration.m_s2(-acc0))
    source.move(lps_qty.Time.s(1), 8)
    source.state_list[-1].acceleration = lps_dynamic.Acceleration(
                lps_qty.Acceleration.m_s2(-acc0), lps_qty.Acceleration.m_s2(acc0))
    source.move(lps_qty.Time.s(1), 8)
    source.state_list[-1].acceleration = lps_dynamic.Acceleration(
                lps_qty.Acceleration.m_s2(acc0), lps_qty.Acceleration.m_s2(acc0))
    source.move(lps_qty.Time.s(1), 8)

    sonar = lps_dynamic.Element()
    omni_sensor.set_base_element(sonar)
    shaded_sensor.set_base_element(sonar)
    sonar.move(lps_qty.Time.s(1), 32)

    signal_in = sin_noise.generate_noise(fs)
    signal_omni = omni_sensor.transduce(signal_in, sin_noise, fs)
    signal_shaded = shaded_sensor.transduce(signal_in, sin_noise, fs)

    n_samples = len(signal_in)
    time = np.arange(n_samples) / fs.get_hz()

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, signal_in)
    plt.title("Original Signal (µPa)")
    plt.ylabel("Pressure (µPa)")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, signal_omni)
    plt.title("After Omni Sensor (V)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, signal_shaded)
    plt.title("After Shaded Sensor (V)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,"sensor_comparison.png"))
    plt.close()

    def get_trajectory(element):
        x = []
        y = []
        for state in element.state_list:
            x.append(state.position.x.get_m())
            y.append(state.position.y.get_m())
        return np.array(x), np.array(y)

    x_src, y_src = get_trajectory(source)
    x_sonar, y_sonar = get_trajectory(sonar)

    plt.figure(figsize=(6, 6))
    plt.plot(x_src, y_src, label='Source Trajectory', color='tab:red')
    plt.plot(x_sonar, y_sonar, label='Sonar Trajectory', color='tab:blue')
    plt.scatter(x_src[0], y_src[0], color='red', marker='o', label='Source Start')
    plt.scatter(x_sonar[0], y_sonar[0], color='blue', marker='x', label='Sonar Start')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajetória dos Elementos na Simulação")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "trajectories.png"))
    plt.close()



if __name__ == "__main__":
    main()
