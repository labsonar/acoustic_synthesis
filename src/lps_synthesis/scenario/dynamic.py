"""
Module for representing the elements and dynamics of the scenario
"""
import typing
import math
import copy

import numpy as np
import matplotlib.pyplot as plt

import lps_utils.quantities as lps_qty


U = typing.TypeVar('U', bound=lps_qty.Quantity)

class Vector(typing.Generic[U]):
    """ Vector representation for space of a quantity. """

    def __init__(self, x: U, y: U, z: typing.Optional[U] = None) -> None:
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def polar3d(cls, magnitude: U, azimuth: lps_qty.Angle, elevation: lps_qty.Angle) -> 'Vector[U]':
        """ Class constructor based on polar representation

        Args:
            magnitude (U): Magnitude of 3D vector
            azimuth (lps_qty.Angle): Angle in XY plane
            elevation (lps_qty.Angle): Angle with Z axis
        """
        x = magnitude * math.sin(elevation.get_rad()) * math.cos(azimuth.get_rad())
        y = magnitude * math.sin(elevation.get_rad()) * math.sin(azimuth.get_rad())
        z = magnitude * math.cos(elevation.get_rad())
        return cls(x=x, y=y, z=z)

    @classmethod
    def polar2d(cls, magnitude: U, azimuth: lps_qty.Angle,
                z: typing.Optional[U] = None) -> 'Vector[U]':
        """ Class constructor based on polar representation in XY plane

        Args:
            magnitude (U): Magnitude of vector in XY plane
            azimuth (lps_qty.Angle): Angle in XY plane
            z (typing.Optional[U], optional): Distance in z axis to reference.
                Defaults to None (2D vector).
        """
        x = magnitude * math.cos(azimuth.get_rad())
        y = magnitude * math.sin(azimuth.get_rad())
        return cls(x=x, y=y, z=z)

    def get_magnitude(self) -> U:
        """ Get magnitude of the vector. """
        if self.z is None:
            return ((self.x**2) + (self.y**2))**0.5
        return ((self.x**2) + (self.y**2) + self.z**2)**0.5

    def get_magnitude_xy(self) -> U:
        """ Get magnitude of the vector in XY plane. """
        return ((self.x**2) + (self.y**2))**0.5

    def get_azimuth(self) -> lps_qty.Bearing:
        """ Get azimuth of the vector (angle in XY plane). """
        if self.x.magnitude == 0 and self.y.magnitude == 0:
            return lps_qty.Bearing.eccw_rad(0)
        return lps_qty.Bearing.eccw_rad(math.atan2(self.y.get(self.x.unity, self.x.prefix),
                                    self.x.get(self.x.unity, self.x.prefix)))

    def get_elevation(self) -> lps_qty.Angle:
        """ Get elevation of the vector (angle in Z axis). """
        if self.z is None:
            return lps_qty.Angle.deg(0)

        magnitude = self.get_magnitude()
        return lps_qty.Angle.rad(math.asin(self.z / magnitude))

    def project(self, direction: lps_qty.Bearing) -> U:
        """ Returns the projection of magnitude onto direction. """
        return math.cos((direction - self.get_azimuth()).get_ccw_rad()) * self.get_magnitude_xy()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self.z is None:
            return f"{self.x}, {self.y}"
        return f"{self.x}, {self.y}, {self.z}"

    def __eq__(self, other: 'Vector[U]') -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other: 'Vector[U]') -> bool:
        return not self == other

    def __gt__(self, other: 'Vector[U]') -> bool:
        return self.get_magnitude() > other.get_magnitude()

    def __lt__(self, other: 'Vector[U]') -> bool:
        return self.get_magnitude() < other.get_magnitude()

    def __ge__(self, other: 'Vector[U]') -> bool:
        return self.get_magnitude() >= other.get_magnitude()

    def __le__(self, other: 'Vector[U]') -> bool:
        return self.get_magnitude() <= other.get_magnitude()

    def __add__(self, other: 'Vector[U]') -> 'Vector[U]':
        z = None
        if self.z is not None and other.z is not None:
            z = self.z + other.z
        elif self.z is not None:
            z = self.z
        elif other.z is not None:
            z = other.z

        return Vector(self.x + other.x, self.y + other.y, z)

    def __sub__(self, other: 'Vector[U]') -> 'Vector[U]':
        z = None
        if self.z is not None and other.z is not None:
            z = self.z - other.z
        elif self.z is not None:
            z = self.z
        elif other.z is not None:
            z = other.z

        return Vector(self.x - other.x, self.y - other.y, z)

    def __mul__(self, other: typing.Union[float, lps_qty.Time]) -> 'Vector[U]':
        if isinstance(other, (int, float)):
            return Vector(self.x * other, self.y * other,
                          self.z * other if self.z is not None else None)
        if isinstance(other, lps_qty.Time):
            return Vector(self.x * other, self.y * other,
                          self.z * other if self.z is not None else None)
        raise TypeError("Invalid operand type for multiplication")

    def __rmul__(self, other: typing.Union[float, lps_qty.Time]) -> 'Vector[U]':
        return self * other

    def __truediv__(self, other: typing.Union[float, lps_qty.Time]) -> 'Vector[U]':
        if isinstance(other, (int, float)):
            return Vector(self.x / other, self.y / other,
                          self.z / other if self.z is not None else None)
        if isinstance(other, lps_qty.Time):
            return Vector(self.x / other, self.y / other,
                          self.z / other if self.z is not None else None)
        else:
            raise TypeError("Invalid operand type for division")

Displacement = Vector[lps_qty.Distance]
Velocity = Vector[lps_qty.Speed]
Acceleration = Vector[lps_qty.Acceleration]

class Point():
    """ Class to represent a point in the globe. """

    def __init__(self,
                 latitude: lps_qty.Latitude = lps_qty.Latitude.rad(0),
                 longitude: lps_qty.Longitude = lps_qty.Longitude.rad(0)) -> None:
        self.latitude = latitude
        self.longitude = longitude

    @classmethod
    def rad(cls, lat: float, lon: float) -> 'Point':
        """ Class constructor in radian. """
        return cls(lps_qty.Latitude.rad(lat), lps_qty.Longitude.rad(lon))

    @classmethod
    def deg(cls, lat: float, lon: float) -> 'Point':
        """ Class constructor in degrees. """
        return cls(lps_qty.Latitude.deg(lat), lps_qty.Longitude.deg(lon))

    def __str__(self) -> str:
        return f'{self.latitude} {self.longitude}'

    def __sub__(self, other: 'Point') -> Displacement:
        if isinstance(other, Point):
            return global_distance(other, self)
        if isinstance(other, Vector):
            return self + (-1 * other)
        raise NotImplementedError(f"{type(self)} - {type(other)} not implemented")

    def __add__(self, other: Displacement) -> Displacement:
        return global_displacement(self, other)

    def __radd__(self, other: Displacement) -> Displacement:
        return self + other


def global_distance(p2: Point, p1: Point) -> Displacement:
    """ Calculate the distance vector between two Point in the globe

    Args:
        p2 (Point): Final point
        p1 (Point): Starting point

    Returns:
        Displacement: Displacement needed to move from p1 to p2
    """
    a = 6378137.0
    f = 1.0/298.257223563
    b = (1.0 - f) * a

    treshold = 1E-12
    max_iter = 200

    _u1 = math.atan((1.0-f) * math.tan(p1.latitude.get_rad()))
    _u2 = math.atan((1.0-f) * math.tan(p2.latitude.get_rad()))

    _l = p2.longitude.get_rad() - p1.longitude.get_rad()

    _lambda = _l
    _lambda_temp = _l

    for _ in range(max_iter):

        t1 = math.cos(_u2) * math.sin(_lambda)
        t2 = math.cos(_u1) * math.sin(_u2) - math.sin(_u1) * math.cos(_u2) * math.cos(_lambda)
        sin_sigma = math.sqrt(math.pow(t1,2) + math.pow(t2,2))
        cos_sigma = math.sin(_u1) * math.sin(_u2) + \
                    math.cos(_u1) * math.cos(_u2) * math.cos(_lambda)
        sigma = math.atan(sin_sigma / cos_sigma)
        sin_alpha = (math.cos(_u1) * math.cos(_u2) * math.sin(_lambda)) / sin_sigma
        cos2_alpha = 1 - math.pow(sin_alpha,2)
        cos_2sigma_m = cos_sigma - (2 * math.sin(_u1) * math.sin(_u2)) / cos2_alpha
        _c = (f/16) * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
        _lambda = _l + (1 - _c) * f * sin_alpha * (sigma + _c * sin_sigma * \
                    (cos_2sigma_m + _c * cos_sigma * (-1 + 2 * math.pow(cos_2sigma_m,2))))

        if math.fabs(_lambda_temp - _lambda) < treshold:
            break

        _lambda_temp = _lambda

    u2 = cos2_alpha * ((math.pow(a,2) - math.pow(b,2)) / math.pow(b,2))
    _a = 1 + u2 / 16384 * (4096 + u2 *( -768 + u2 * (320 - 175 * u2)))
    _b = u2 / 1024 * (256 + u2 *( -128 + u2 * (74.0 - 47 * u2)))
    delta_sigma = _b * sin_sigma * (cos_2sigma_m + _b / 4* (cos_sigma * \
            (-1 + 2 * math.pow(cos_2sigma_m,2)) - _b / 6 * cos_2sigma_m * \
            (-3 + 4 * math.pow(sin_sigma,2)) * (-3 + 4 * math.pow(cos_2sigma_m,2))))
    s = b * _a * (sigma - delta_sigma)

    t1 = math.cos(_u2) * math.sin(_lambda)
    t2 = math.cos(_u1) * math.sin(_u2) - math.sin(_u1) * math.cos(_u2) * math.cos(_lambda)

    alpha1 = math.atan2(t1,t2)

    return Displacement.polar2d(lps_qty.Distance.m(s), lps_qty.Angle.rad(alpha1))

def global_displacement(p: Point, d: Displacement) -> Point:
    """ Calculate the Point in globe after a displacement

    Args:
        p (Point): starting point
        d (Displacement): desired displacement

    Returns:
        Point: final point
    """
    a = 6378137.0
    f = 1.0/298.257223563
    b = (1.0 - f) * a

    treshold_sigma = 1E-15
    max_iter = 200

    u1 = math.atan((1.0 - f) * math.tan(p.latitude.get_rad()))
    sigma1 = math.atan2(math.tan(u1),math.cos(d.get_azimuth().get_rad()))

    sin_alpha = math.cos(u1) * math.sin(d.get_azimuth().get_rad())
    cos2_alpha = 1 - math.pow(sin_alpha,2)
    u2 = cos2_alpha * ((math.pow(a,2) - math.pow(b,2)) / math.pow(b,2))
    _a = 1 + u2 / 16384 * (4096 + u2 *( -768 + u2 * (320 - 175 * u2)))
    _b = u2 / 1024 * (256 + u2 *( -128 + u2 * (74.0 - 47 * u2)))

    sigma = d.get_magnitude().get_m() / (b * _a)
    sigma_temp = sigma

    for _ in range(max_iter):
        sigma_m = 2 * sigma1 + sigma
        delta_sigma = _b * math.sin(sigma) * (math.cos(sigma_m) + \
                      _b / 4* (math.cos(sigma) * (-1 + 2 * math.pow(math.cos(sigma_m),2)) - \
                      _b / 6 * math.cos(sigma_m) * (-3 + 4 * math.pow(math.sin(sigma),2)) * \
                            (-3 + 4 * math.pow(math.cos(sigma_m),2))))

        sigma = d.get_magnitude().get_m() / (b * _a) + delta_sigma
        if math.fabs(sigma_temp - sigma) < treshold_sigma:
            break
        sigma_temp = sigma

    phi2 = math.atan2(
        math.sin(u1) * math.cos(sigma) + \
            math.cos(u1) * math.sin(sigma) * math.cos(d.get_azimuth().get_rad()),
        (1 - f) * math.sqrt(math.pow(sin_alpha,2) + math.pow(math.sin(u1) * math.sin(sigma) - \
            math.cos(u1) * math.cos(sigma) * math.cos(d.get_azimuth().get_rad()),2))
        )

    _lambda = math.atan2(math.sin(sigma) * math.sin(d.get_azimuth().get_rad()),
                math.cos(u1) * math.cos(sigma) - \
                math.sin(u1) * math.sin(sigma) * math.cos(d.get_azimuth().get_rad()))

    _c = (f/16) * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
    _l = _lambda - (1 - _c) * f * sin_alpha * (sigma + _c * math.sin(sigma) * (math.cos(sigma_m) + \
                            _c * math.cos(sigma) * (-1 + 2 * math.pow(math.cos(sigma_m),2))))

    _l2 = _l + p.longitude.get_rad()

    return Point.rad(phi2,_l2)


class State():
    """ Class to represent any element dynamic state in the simulation """

    def __init__(self,
                position: Displacement = Displacement(lps_qty.Distance.m(0),
                                        lps_qty.Distance.m(0),
                                        lps_qty.Distance.m(0)),
                velocity: Velocity = Velocity(lps_qty.Speed.m_s(0), lps_qty.Speed.m_s(0)),
                acceleration: Acceleration = Acceleration(lps_qty.Acceleration.m_s2(0),
                                                          lps_qty.Acceleration.m_s2(0)),
                max_speed: lps_qty.Speed = lps_qty.Speed.kt(50)) -> None:
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.max_speed = max_speed

    def estimate(self, dt: lps_qty.Time) -> 'State':
        """ Estimate state after a delta time """

        position=self.position + self.velocity*dt + 0.5*self.acceleration*dt*dt
        velocity = self.velocity + self.acceleration*dt
        acceleration=self.acceleration

        if self.max_speed is not None and velocity.get_magnitude() > self.max_speed:
            velocity *= self.max_speed/velocity.get_magnitude()
            acceleration = Acceleration(lps_qty.Acceleration.m_s2(0),
                                        lps_qty.Acceleration.m_s2(0))

        return State(position=position, velocity=velocity,
                     acceleration=acceleration, max_speed=self.max_speed)

    def __str__(self) -> str:
        return  f"[{self.position}], [{self.velocity}], [{self.acceleration}]"

    def get_relative_speed(self, other_state: 'State') -> lps_qty.Speed:
        """ Get the approach speed of another state. """
        diff = other_state.position - self.position
        return self.velocity.project(diff.get_azimuth())

class Element():
    """ Class to represent any element that moves in the simulation """

    def __init__(self,
                initial_state: State = State()) -> None:
        self.ref_state = initial_state.estimate(lps_qty.Time.s(0))
        self.reset()

    def reset(self, initial_state: State = None) -> None:
        """ Resets the element to prepare for simulation. """
        if initial_state is None:
            initial_state = self.ref_state
        else:
            self.ref_state = initial_state

        self.state_list = []
        self.state_list.append(initial_state)
        self.step_interval = []

    def move(self, step_interval: lps_qty.Speed, n_steps: int = 1) -> typing.List[State]:
        """ Calculates the state of this element throughout a simulation."""

        for _ in range(n_steps):
            new_state = self.state_list[-1].estimate(step_interval)
            self.state_list.append(new_state)
            self.step_interval.append(step_interval)

        return self.state_list

    def __str__(self) -> str:
        return str(self.state_list[0])

    def __getitem__(self, step: int) -> State:
        return self.state_list[step]

    def get_depth(self) -> lps_qty.Distance:
        """ Return the starting depth of the element. """
        return self.ref_state.position.z

    def get_simulated_steps(self) -> typing.Iterator[typing.Tuple[State, lps_qty.Time]]:
        """ Return the associated states and simulation steps. """
        return zip(self.state_list, self.step_interval)

    def get_n_steps(self) -> int:
        """ Return the number of simulation steps. """
        return len(self.state_list)

    @staticmethod
    def plot_trajectories(elements : typing.List['Element'], filename: str):
        """
        Plots the trajectory of multiple elements.

        Args:
        elements (list): List of elements.
        filename (str): Name of the output file.
        """

        def get_trajectory(element):
            x = [state.position.x.get_m() for state in element.state_list]
            y = [state.position.y.get_m() for state in element.state_list]
            return np.array(x), np.array(y)

        plt.figure(figsize=(8, 8))
        colors = plt.cm.get_cmap('tab10', len(elements))

        for i, element in enumerate(elements):
            x, y = get_trajectory(element)

            try:
                label = element.get_id()
            except AttributeError:
                label = f"Element {i}"

            plt.plot(x, y, label=label, color=colors(i))
            plt.scatter(x[-1], y[-1], color=colors(i), marker='o')

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def relative_distance_plot(ref_element: 'Element',
                               elements : typing.List['Element'],
                               filename: str) -> None:
        """ Make plots with relative distances between ref_element and elements. """
        n_steps = ref_element.get_n_steps()

        dt = np.array([s.get_s() for s in ref_element.step_interval])
        t = np.concatenate(([0.0], np.cumsum(dt)))

        for i, element in enumerate(elements):

            try:
                label = element.get_id()
            except AttributeError:
                label = f"Element {i}"

            dist = []
            for step_i in range(n_steps):
                diff = element[step_i].position - ref_element[step_i].position
                dist.append(diff.get_magnitude_xy().get_km())

            plt.plot(t, dist, label=label)

        plt.xlabel('Time (seconds)')
        plt.ylabel('Distance (km)')
        plt.legend()
        plt.savefig(filename)
        plt.close()


class RelativeElement():
    """ Class to represent any element that moves in the simulation """

    def __init__(self, rel_position: Displacement = \
                    Displacement(lps_qty.Distance.m(0), lps_qty.Distance.m(0))) -> None:
        self.rel_position = rel_position
        self.ref_element = None

    def set_base_element(self, element: Element):
        """ Set the element this element is relative to. """
        self.ref_element = element

    # def __getitem__(self, step: int) -> State:
    #     self.check()
    #     state = copy.deepcopy(self.ref_element[step])
    #     state.position = state.position + self.rel_position
    #     return state

    def __getitem__(self, step: int) -> State:
        self.check()
        state = copy.deepcopy(self.ref_element[step])

        heading = state.velocity.get_azimuth().get_eccw_rad()

        rotated_x = self.rel_position.x * math.cos(heading) - \
                        self.rel_position.y * math.sin(heading)
        rotated_y = self.rel_position.x * math.sin(heading) + \
                        self.rel_position.y * math.cos(heading)

        state.position = state.position + Displacement(rotated_x, rotated_y, self.rel_position.z)

        return state


    def get_depth(self) -> lps_qty.Distance:
        """ Return the starting depth of the element. """
        self.check()
        return self.ref_element.get_depth()

    def check(self):
        """ Check if reference element is set, If not raise UnboundLocalError. """
        if self is None:
            raise UnboundLocalError("Relative item must be set before use")
