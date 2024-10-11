""" Example of use of propagation module
"""
import time

import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.acoustical_channel as lps_channel

desc = lps_channel.Description()
desc.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))
desc.add(lps_qty.Distance.m(50), lps_layer.BottomType.CHALK)

start_time = time.time()
channel = lps_channel.Channel(
                description = desc,
                source_depth = [lps_qty.Distance.m(5), lps_qty.Distance.m(15)],
                sensor_depth = lps_qty.Distance.m(40),
                max_distance = lps_qty.Distance.m(400),
                max_distance_points = 100,
                sample_frequency = lps_qty.Frequency.khz(16),
                frequency_range = (lps_qty.Frequency.khz(5.2), lps_qty.Frequency.khz(5.5)),
                temp_dir = './result/propagation')
end_time = time.time()
print("elapsed_time: ", end_time-start_time)

channel.export_h_f('./result/h_f_.png')
channel.export_h_f('./result/h_f_2.png', source_id=1)
channel.export_h_t_tau('./result/h_t_tau_.png')
channel.export_h_t_plots('./result/h_t.png')
