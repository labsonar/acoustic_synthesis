"""
Module for representing the elements and dynamics of the scenario
"""
import typing
import math

import lps_utils.quantities as lps_qty


U = typing.TypeVar('U', bound=lps_qty.Quantity)

class Vector(typing.Generic[U]):
    """ Vector representation for space of a quantity. """

    def __init__(self, x: U, y: U, z: typing.Optional[U] = None) -> None:
        self._x = x
        self._y = y
        self._z = z

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
        if self._z is None:
            return (self._x**2 + self._y**2)**0.5
        return (self._x**2 + self._y**2 + self._z**2)**0.5

    def get_azimuth(self) -> lps_qty.Angle:
        """ Get azimuth of the vector (angle in XY plane). """
        if self._x.magnitude == 0 and self._y.magnitude == 0:
            return 0.0
        return lps_qty.Angle.rad(math.atan2(self._y.get(self._x.unity, self._x.prefix),
                                    self._x.get(self._x.unity, self._x.prefix)))

    def get_elevation(self) -> lps_qty.Angle:
        """ Get elevation of the vector (angle in Z axis). """
        if self._z is None:
            return lps_qty.Angle.deg(0)

        magnitude = self.get_magnitude()
        return lps_qty.Angle.rad(math.asin(self._z / magnitude))

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self._z is None:
            return f"{self._x}, {self._y}"
        return f"{self._x}, {self._y}, {self._z}"

    def __eq__(self, other: 'Vector[U]') -> bool:
        return self._x == other._x and self._y == other._y and self._z == other._z

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
        if self._z is not None and other._z is not None:
            z = self._z + other._z
        elif self._z is not None:
            z = self._z
        elif other._z is not None:
            z = other._z

        return Vector(self._x + other._x, self._y + other._y, z)

    def __sub__(self, other: 'Vector[U]') -> 'Vector[U]':
        z = None
        if self._z is not None and other._z is not None:
            z = self._z - other._z
        elif self._z is not None:
            z = self._z
        elif other._z is not None:
            z = other._z

        return Vector(self._x - other._x, self._y - other._y, z)

    def __mul__(self, other: typing.Union[float, lps_qty.Time]) -> 'Vector[U]':
        if isinstance(other, (int, float)):
            return Vector(self._x * other, self._y * other,
                          self._z * other if self._z is not None else None)
        if isinstance(other, lps_qty.Time):
            return Vector(self._x * other, self._y * other,
                          self._z * other if self._z is not None else None)
        raise TypeError("Invalid operand type for multiplication")

    def __rmul__(self, other: typing.Union[float, lps_qty.Time]) -> 'Vector[U]':
        return self * other

    def __truediv__(self, other: typing.Union[float, lps_qty.Time]) -> 'Vector[U]':
        if isinstance(other, (int, float)):
            return Vector(self._x / other, self._y / other,
                          self._z / other if self._z is not None else None)
        if isinstance(other, lps_qty.Time):
            return Vector(self._x / other, self._y / other,
                          self._z / other if self._z is not None else None)
        else:
            raise TypeError("Invalid operand type for division")

Displacement = Vector[lps_qty.Distance]
Velocity = Vector[lps_qty.Speed]


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
