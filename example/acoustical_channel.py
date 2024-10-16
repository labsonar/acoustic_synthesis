import matplotlib.pyplot as plt

import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.acoustical_channel as lps_channel

num_profiles: int = 5

plt.figure(figsize=(8, 6))

for i in range(num_profiles):
    description = lps_channel.Description.get_random()

    depths = []
    speeds = []
    for depth, layer in description:
        if isinstance(layer, lps_layer.Water):
            depths.append(depth.get_m())
            speeds.append(layer.get_compressional_speed().get_m_s())
        if isinstance(layer, lps_layer.SeabedType):
            print(layer)

    plt.plot(speeds, depths, label=f'Perfil {i+1}')


plt.gca().invert_yaxis()
plt.xlabel("Velocidade do Som (m/s)")
plt.ylabel("Profundidade (m)")
plt.title(f"Perfis Acústicos Aleatórios ({num_profiles} perfis)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./result/acoustical_channel.png")
