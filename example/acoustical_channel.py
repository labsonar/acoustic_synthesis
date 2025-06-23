"""Simple acoustical channel test. """
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.channel_description as lps_channel

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
        if isinstance(layer, lps_layer.Seabed):
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


print('description: ', description)

print('20: ', description.get_speed_at(lps_qty.Distance.m(20)))
print('50: ', description.get_speed_at(lps_qty.Distance.m(50)))
print('100: ', description.get_speed_at(lps_qty.Distance.m(100)))
