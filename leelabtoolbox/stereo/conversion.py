"""some conversion utilities"""
from __future__ import division, unicode_literals, print_function, absolute_import
import numpy as np


# the conversion between spherical and 3D Cartesian follows convention in
# <https://github.com/astropy/astropy/blob/v1.3.x/astropy/coordinates/representation.py#L956-L987>
# as well as MATLAB https://www.mathworks.com/help/matlab/ref/sph2cart.html

def _check_sph_range(d, lat, lon, convention):
    assert np.all(d > 0)  # simply don't think about ambiguous case.
    assert np.all(np.logical_and(lat >= -np.pi / 2, lat <= np.pi / 2))
    if convention == 'standard':
        assert np.all(np.logical_and(lon >= 0, lon <= np.pi * 2))
    elif convention == 'retina2':
        assert np.all(np.logical_and(lon >= -np.pi, lon <= np.pi))
    else:
        raise ValueError('unsupported convention!')


def _check_all_data_finite(*arrays):
    for a in arrays:
        np.isfinite(a).all()


def _check_input_same_shape(*arrays):
    new_data = []
    assert len(arrays) >= 1
    scalar_flag = np.isscalar(arrays[0])
    if scalar_flag:
        for a in arrays[1:]:
            assert np.isscalar(a)
        return arrays, scalar_flag
    else:
        array0 = np.asarray(arrays[0])
        data_shape = array0.shape
        new_data.append(array0)
        for a in arrays[1:]:
            new_array = np.asarray(a)
            assert new_array.shape == data_shape
            new_data.append(new_array)
        return new_data, scalar_flag


def sph2cart(d, lat, lon, convention='standard'):
    """ spherical to cartesian.


    :param d: distance (radius). must be positive.
    :param lat: latitude should be between -pi/2 (south) to pi/2 (north).
           also called elevation; pi/2-elevation gives inclination, see wiki.
    :param lon: longitude (azimuth). between 0 to 360 degrees (2 * pi) for standard, and [-pi, pi] for retina2

    all three must have exactly the same shape.

    :return: x, y, z
    """

    (d, lat, lon), _ = _check_input_same_shape(d, lat, lon)
    # check their ranges. these are based on astropy.
    _check_all_data_finite(d, lat, lon)
    _check_sph_range(d, lat, lon, convention)

    if convention == 'standard':
        # same implementation as in MATLAB and astropy, where x, y axis lie on equator, and z cross north and south poles.
        # check <math_sphcart.png>, copied from MATLAB page on sph2cart.
        # also check wiki <https://en.wikipedia.org/wiki/Spherical_coordinate_system>
        x = d * np.cos(lat) * np.cos(lon)
        y = d * np.cos(lat) * np.sin(lon)
        z = d * np.sin(lat)
    elif convention == 'retina2':
        # my old convention that's easy to work with stereo dataset, insipred by OpenGL coordinate system
        # where x is right, y is top, and negative z is away from camera.
        # check <https://github.com/leelabcnbc/personal_library_suite#stereo>
        # there, azimuth is longitude, and elevation is latitude.

        y = d * np.sin(lat)
        commmon_ = d * np.cos(lat)
        x = -commmon_ * np.sin(lon)
        z = -commmon_ * np.cos(lon)
    else:
        raise ValueError('unsupported convention!')
    _check_all_data_finite(x, y, z)

    return x, y, z


def cart2sph(x, y, z, convention='standard'):
    (x, y, z), scalar_flag = _check_input_same_shape(x, y, z)
    _check_all_data_finite(x, y, z)

    if convention == 'standard':
        s = np.hypot(x, y)
        r = np.hypot(s, z)

        lon = np.arctan2(y, x)
        lat = np.arctan2(z, s)

        # I need to fix the conventions.
        # lat should be good.
        # I need to fix lon to be in (0, 2pi)
        _check_sph_range(r, lat, lon, convention='retina2')
        # then fix lon
        if scalar_flag:
            if lon < 0:
                lon += np.pi * 2
        else:
            lon[lon < 0] += np.pi * 2
    elif convention == 'retina2':
        # this doesn't need conversion, as my definition works naturally with atan2
        s = np.hypot(x, z)
        r = np.hypot(s, y)
        lat = np.arctan2(y, s)
        lon = np.arctan2(-x, -z)
    else:
        raise ValueError('unsupported convention!')
    _check_sph_range(r, lat, lon, convention)
    _check_all_data_finite(r, lat, lon)
    return r, lat, lon
