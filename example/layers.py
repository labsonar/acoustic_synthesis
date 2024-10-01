import time
import numpy as np

import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.acoustical_channel as lps_channel
import lps_synthesis.propagation.oases as lps_oases

channel = lps_channel.Description()
channel.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1534))
channel.add(lps_qty.Distance.m(5), lps_qty.Speed.m_s(1536))
channel.add(lps_qty.Distance.m(50), lps_qty.Speed.m_s(1536))
channel.add(lps_qty.Distance.m(80), lps_qty.Speed.m_s(1530))
channel.add(lps_qty.Distance.m(100), lps_qty.Speed.m_s(1528))
channel.add(lps_qty.Distance.m(400), lps_layer.BottomType.BASALT)

print("#### oases_format ####")
print(channel.to_oases_format())

print("#### as str ####")
print(channel)

print("#### Looping in layers ####")
for depth, layer in channel:
    print(depth, layer)

channel.export_ssp("./result/ssp.png")