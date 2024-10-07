""" Example of use of propagation module
"""
import time

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.acoustical_channel as lps_channel

desc = lps_channel.Description()
desc.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1530))
desc.add(lps_qty.Distance.m(10), lps_qty.Speed.m_s(1530))
desc.add(lps_qty.Distance.m(30), lps_qty.Speed.m_s(1480))
desc.add(lps_qty.Distance.m(50), lps_qty.Speed.m_s(1475))
desc.add(lps_qty.Distance.m(50), lps_layer.BottomType.CLAY)

start_time = time.time()
channel = lps_channel.Channel(
                description = desc,
                source_depth = lps_qty.Distance.m(5),
                sensor_depth = lps_qty.Distance.m(40),
                max_distance = lps_qty.Distance.m(1000),
                distance_points = 512,
                sample_frequency = lps_qty.Frequency.khz(16),
                n_fft = 128,
                temp_dir = './result/propagation')
end_time = time.time()
print("elapsed_time: ", end_time-start_time)

channel.export_h_f('./result/h_f_.png')
