"""Oases test
"""
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.oases as oases
import lps_synthesis.propagation.acoustical_channel as lps_channel

desc = lps_channel.Description()
desc.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))
desc.add(lps_qty.Distance.m(50), lps_layer.BottomType.CHALK)

h_f_tau, h_t_tau, z, r, f, t = oases.estimate_transfer_function(
    description = desc,
    # source_depth = [lps_qty.Distance.m(5), lps_qty.Distance.m(10), lps_qty.Distance.m(8)],
    source_depth = [lps_qty.Distance.m(5), lps_qty.Distance.m(10), lps_qty.Distance.m(15)],
    sensor_depth = lps_qty.Distance.m(40),
    max_distance = lps_qty.Distance.m(400),
    max_distance_points = 50,
    sample_frequency = lps_qty.Frequency.khz(16),
    # frequency_range = (lps_qty.Frequency.khz(5.2), lps_qty.Frequency.khz(5.5)),
    filename = "./result/test.dat")

# print("[z] ", z[0], " -> ", z[-1], ": ", len(z))
# print("[r] ", r[0], " -> ", r[-1], ": ", len(r))
# print("[f] ", f[0], " -> ", f[-1], ": ", len(f))
# print("[t] ", t[0], " -> ", t[-1], ": ", len(t))

plt.imshow(abs(h_f_tau[0,:,:]), aspect='auto', cmap='jet', extent=[
            f[0].get_khz(),
            f[-1].get_khz(),
            r[0].get_km(),
            r[-1].get_km()]
)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Distance (km)")
plt.colorbar()
plt.tight_layout()
plt.savefig("./result/h_f_tau.png")
plt.close()

plt.imshow(abs(h_t_tau[0,:,:]), aspect='auto', cmap='jet', interpolation='none',
       extent=[
            t[0].get_s(),
            t[-1].get_s(),
            r[-1].get_m(),
            r[0].get_m()]
    )

plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.colorbar()
plt.tight_layout()
plt.savefig("./result/h_t_tau.png")
plt.close()


times = [t_i.get_s() for t_i in t]
labels = []
for r_i in range(0, len(r), 1):
    plt.plot(times, abs(h_t_tau[0,r_i,:]))
    labels.append(str(r[r_i]))

plt.legend(labels)
plt.savefig("./result/h_t.png")
plt.close()
