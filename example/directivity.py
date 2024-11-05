import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic
import lps_synthesis.scenario.sonar as lps_sonar

# Configurações para o sonar e sensores
n_staves = 32
radius = lps_qty.Distance.m(10)
sensitivity = lps_qty.Sensitivity.db_v_p_upa(-150)  # Exemplo de sensibilidade; ajuste conforme necessário

# Inicialização do sonar cilíndrico
sonar = lps_sonar.Sonar.cylindrical(n_staves=n_staves, radius=radius, sensitivity=sensitivity)
sonar[0].velocity.y = lps_qty.Speed.kt(1)

# Função para calcular o ganho direcional em diferentes ângulos
def plot_polar_gain(sensor, ax, title, distance=lps_qty.Distance.m(200)):
    angles = np.linspace(0, 2 * np.pi, 360)

    gains = []
    for angle in angles:
        print("angle: ", angle)
        x_position = distance * np.cos(angle)
        y_position = distance * np.sin(angle)
        source_position = lps_dynamic.Displacement(x_position, y_position)

        gain = sensor.direction_gain(0, source_position)
        gains.append(gain)

    ax.plot(angles, gains)
    ax.set_title(title, va='bottom')

# Plota os gráficos polares para sensores 0 e 8 antes da rotação
fig, axs = plt.subplots(2, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 8))

plot_polar_gain(sonar.sensors[0], axs[0, 0], "Sensor 0 (Antes da rotação)")
plot_polar_gain(sonar.sensors[8], axs[0, 1], "Sensor 8 (Antes da rotação)")

# Define uma nova velocidade angular para o sonar e simula uma rotação
sonar[0].velocity = lps_dynamic.Velocity(lps_qty.Speed.kt(5), lps_qty.Speed.kt(5))

# Plota os gráficos polares para sensores 0 e 8 após a rotação
plot_polar_gain(sonar.sensors[0], axs[1, 0], "Sensor 0 (Após rotação)")
plot_polar_gain(sonar.sensors[8], axs[1, 1], "Sensor 8 (Após rotação)")

# Ajusta layout e salva o gráfico
plt.tight_layout()
plt.savefig('./result/directivity.png')
plt.show()
