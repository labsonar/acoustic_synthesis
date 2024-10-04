""" Test for dynamic module
"""
import lps_utils.quantities as lps_qty
import lps_synthesis.scenario.dynamic as lps_dynamic

print("\n======= Vector =======")
d2 = lps_dynamic.Displacement(lps_qty.Distance.m(100), lps_qty.Distance.m(100))
d3 = lps_dynamic.Displacement(lps_qty.Distance.m(200),
                              lps_qty.Distance.m(200),
                              lps_qty.Distance.m(200))

print('d2: ', d2)
print('d2: ', d2.get_magnitude(), " ", d2.get_azimuth(), " ", d2.get_elevation())
print('d3: ', d3)
print('d3: ', d3.get_magnitude(), " ", d3.get_azimuth(), " ", d3.get_elevation())

print('d2 + d2: ', d2 + d2)
print('d2 + d3: ', d2 + d3)
print('d3 + d3: ', d3 + d3)
print('d2 - d2: ', d2 - d2)
print('d2 - d3: ', d2 - d3)
print('d3 - d3: ', d3 - d3)

vel = lps_dynamic.Velocity(lps_qty.Speed.kt(10), lps_qty.Speed.kt(20))
print('vel: ', vel)

dt = lps_qty.Time.s(10)
print('d2/dt: ', d2/dt)
print('vel + d2/dt: ', vel + d2/dt)


print("\n======= Point =======")
p1 = lps_dynamic.Point.deg(-22.874828, -43.1333848)
p2 = lps_dynamic.Point.deg(-22.5, -43.0)

print('p1: ', p1)
print('p2: ', p2)
print('p2 - p1: ', p2 - p1)
print('p1 + d2: ', p1 + d2)

print('(p1 - d2) - p1: ', (p1 - d2) - p1)
