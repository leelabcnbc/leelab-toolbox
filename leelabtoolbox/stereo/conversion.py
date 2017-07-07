"""some conversion utilities

overall, math in this module follows those in Chapter 7 of Binocular Vision and Stereopsis (Oxford Psychology Series),
by Ian P. Howard, Brian J. Rogers, where gun turret coordinate system is recommended (pp. 240, Summary).
gun turret model is same as the usual spherical coordinate system.
"""
from __future__ import division, unicode_literals, print_function, absolute_import
import numpy as np
from collections import namedtuple
from transforms3d import euler

# the conversion between spherical and 3D Cartesian follows convention in
# <https://github.com/astropy/astropy/blob/v1.3.x/astropy/coordinates/representation.py#L956-L987>
# as well as MATLAB https://www.mathworks.com/help/matlab/ref/sph2cart.html

CoordinatesCart = namedtuple('CoordinatesCart', ('x', 'y', 'z'))
CoordinatesSph = namedtuple('CoordinatesSph', ('d', 'lat', 'lon'))


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

    return CoordinatesCart(x, y, z)


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
    return CoordinatesSph(r, lat, lon)


def _eye_locations(fixation_point, ipd, debug=False):
    """ a rewrite of rotate_eye_position in personal_library_suite

    :param fixation_point:
    :param ipd:
    :param debug: to test if simplifying the rotation matrix equation or not will help.
    :return:
    """
    f_sph_coord = cart2sph(*fixation_point, convention='retina2')
    # then compute rotation matrix,
    # first rotate along y, and then (new) x.
    # the second should be unnecessary.
    eye_l = np.array([-ipd / 2.0, 0, 0], dtype=np.float64)
    eye_r = np.array([ipd / 2.0, 0, 0], dtype=np.float64)

    # for this, you use r, relative axis
    # for transformation_fixation, you use static (s) one.
    rotation_matrix_head = euler.euler2mat(f_sph_coord.lon, f_sph_coord.lat if debug else 0, 0, axes='ryxz')

    result = np.matmul(rotation_matrix_head, eye_l), np.matmul(rotation_matrix_head, eye_r)
    _check_all_data_finite(*result)
    return result


def _transformation_fixation_legacy(fixation_point, eye_location, return_t=False):
    """ a rewrite of transformation_fixation in personal_library_suite

    :param fixation_point:
    :param eye_location:
    :return:
    """
    assert fixation_point.shape == eye_location.shape == (3,)
    assert eye_location[1] == 0, 'eye must be in XoZ plane'
    # angles.lon and angles.lat gives lon and lat of the fixation point, after translating (no rotating yet)
    # the original system so that eye is at (0,0,0).
    angles = cart2sph(*(fixation_point - eye_location), convention='retina2')
    # then we need to reversely apply transformation on points so that fixation point will be directly in the
    # direction of (0, 0, -1).
    if return_t:
        return euler.euler2mat(-angles.lon, -angles.lat, 0, axes='syxz'), -eye_location
    else:
        return euler.euler2mat(-angles.lon, -angles.lat, 0, axes='syxz')


def _transformation_fixation(fixation_point, eye_location, return_t=False):
    """ correct version.

    :param fixation_point:
    :param eye_location:
    :return:
    """
    assert fixation_point.shape == eye_location.shape == (3,)
    assert eye_location[1] == 0, 'eye must be in XoZ plane'
    # angles.lon and angles.lat gives lon and lat of the fixation point, after translating (no rotating yet)
    # the original system so that eye is at (0,0,0).
    angles_central_eye = cart2sph(*fixation_point, convention='retina2')
    # then we need to reversely apply transformation on points so that fixation point will be directly in the
    # direction of (0, 0, -1).
    first_matrix = euler.euler2mat(-angles_central_eye.lon, -angles_central_eye.lat, 0, axes='syxz')
    # then we do another one, changing from the central eye coordinate to left, right eye coordinate.
    eye_location_in_new_coord = np.matmul(first_matrix, eye_location)
    # then do translation, and then rotation to get the correct coordinates.
    move_one = -eye_location_in_new_coord
    # print(move_one)
    # then another rotation around y axis.
    fixation_point_in_new_coord = np.matmul(first_matrix, fixation_point)
    # print(fixation_point_in_new_coord)

    angles_this_eye = cart2sph(*(fixation_point_in_new_coord + move_one), convention='retina2')
    # it should be the case that
    assert abs(angles_this_eye.lat) < 1e-4, "should be on XoZ plane"
    second_matrix = euler.euler2mat(-angles_this_eye.lon, 0, 0, axes='syxz')

    # combine them into a 3x4 matrix.
    assert first_matrix.shape == second_matrix.shape == (3, 3)
    assert move_one.shape == (3,)
    first_matrix_big = np.eye(4, dtype=np.float64)
    second_matrix_big = np.eye(4, dtype=np.float64)
    translation_big = np.eye(4, dtype=np.float64)
    first_matrix_big[:3, :3] = first_matrix
    second_matrix_big[:3, :3] = second_matrix
    translation_big[:3, 3] = move_one
    overall_matrix = np.matmul(second_matrix_big, np.matmul(translation_big, first_matrix_big))
    assert overall_matrix.shape == (4, 4)
    # and last row should be pretty close to (0,0,0,1)
    assert np.array_equal(overall_matrix[3], np.array([0, 0, 0, 1]))

    # not sure if R is really a rotation matrix. but whatever.
    rotation_overall = overall_matrix[:3, :3]
    t_overall = overall_matrix[:3, 3]

    if return_t:
        return rotation_overall, t_overall
    else:
        return rotation_overall


def _trans_apply(x, R, T, pre, scalar):
    # return R(x+T) or Rx + T
    # this R may not be a rotation.
    if not scalar:
        if pre:
            return np.matmul(x.T + T, R.T).T
        else:
            return (np.matmul(x.T, R.T) + T).T
    else:
        if pre:
            return np.matmul(R, x + T)
        else:
            return np.matmul(R, x) + T


def cart2disparity(x, y, z, fixation_point, infinite_fixation=True, ipd=0.065, debug=False, return_xyz=False,
                   legacy=False):
    """ a rewrite of cart2disparity in personal_library_suite.

    everything assumes retina2 XYZ coordinate system convention

    given fixation point from a certain latitude and longtitude,
    this function assumes that the observer rotate its head (camera), first horizontally (in XOZ plane, around Y axis)
    and then vertically (around X axis, if we also rotate the axes in the first rotation).

    :param x: same as x in cart2sph
    :param y: same as y in cart2sph
    :param z: same as z in cart2sph
    :param fixation_point: (3,) array of fixation point (or just giving the direction of eyesight, if
           infinite_fixation is True.
    :param infinite_fixation: whether fixate on fixation_point on fixate in the direction of it.
    :param ipd: interpupillary distance, default to 0.065m, as in "Disparity statistics in natural scenes"
                by Yang Liu; Alan C. Bovik; Lawrence K. Cormack, doi:10.1167/8.11.19


    definition of disparity here is same as that in above paper. Check Fig. 1 of that paper.
    negative disparity is also called crossed disparity.
    positive is called uncrossed.
    :return:
    """

    (x, y, z), scalar_flag = _check_input_same_shape(x, y, z)
    fixation_point = np.asarray(fixation_point, dtype=np.float64)
    assert fixation_point.shape == (3,)
    _check_all_data_finite(x, y, z, fixation_point)
    xyz_all = np.asarray([x, y, z])

    # first, get the locations of two eyes, in the original XYZ coordinate system
    eye_l_new, eye_r_new = _eye_locations(fixation_point, ipd)

    if legacy:
        trans_f = _transformation_fixation_legacy
        pre = True
    else:
        trans_f = _transformation_fixation
        pre = False

    if not infinite_fixation:
        rotation_l, trans_l = trans_f(fixation_point, eye_l_new, return_t=True)
        rotation_r, trans_r = trans_f(fixation_point, eye_r_new, return_t=True)
    else:
        rotation_l = trans_f(fixation_point, np.array([0, 0, 0], dtype=np.float64))
        rotation_r = rotation_l
        if not legacy:
            trans_l = np.array([ipd / 2, 0, 0], dtype=np.float64)
            trans_r = np.array([-ipd / 2, 0, 0], dtype=np.float64)
        else:
            trans_l = -eye_l_new
            trans_r = -eye_r_new
    if not debug:
        xyz_all_l = _trans_apply(xyz_all, rotation_l, trans_l, pre=pre, scalar=scalar_flag)
        xyz_all_r = _trans_apply(xyz_all, rotation_r, trans_r, pre=pre, scalar=scalar_flag)
    else:
        # well, let's do it one by one...
        shape_later = xyz_all.shape[1:]
        assert scalar_flag == (shape_later == ())
        if scalar_flag:
            xyz_all_l = _trans_apply(xyz_all, rotation_l, trans_l, pre=pre, scalar=True)
            xyz_all_r = _trans_apply(xyz_all, rotation_r, trans_r, pre=pre, scalar=True)
        else:
            xyz_all_l = np.empty((3,) + shape_later, dtype=np.float64)
            xyz_all_r = np.empty((3,) + shape_later, dtype=np.float64)
            # then fill one by one
            # use ndenumerate to generate the index.
            # might not be optimal, but since this is debug, who cares.
            # correctness suffices.

            for index, _ in np.ndenumerate(np.empty(shape_later)):
                index_this = (slice(None),) + index
                xyz_all_l[index_this] = _trans_apply(xyz_all[index_this], rotation_l, trans_l, pre=pre, scalar=True)
                xyz_all_r[index_this] = _trans_apply(xyz_all[index_this], rotation_r, trans_r, pre=pre, scalar=True)

    assert xyz_all_l.shape == xyz_all_r.shape == xyz_all.shape
    _check_all_data_finite(xyz_all_l, xyz_all_r)

    # then compute disparity
    sph_l = cart2sph(*xyz_all_l, convention='retina2')
    sph_r = cart2sph(*xyz_all_r, convention='retina2')
    # left minus right.
    # using this, positive means far, negative means near. (at least this is true for points in front of you;).
    # both (-pi,pi), so in theory can range from  -2pi - 2*pi.
    # I will normalize it with in (-pi, pi)
    disparity = sph_l.lon - sph_r.lon

    assert np.isscalar(disparity) == scalar_flag
    # for -pi and +pi case, we normalize them to +pi. (this extreme value will never happen in reality I guess!)
    if scalar_flag:
        if disparity < -np.pi:
            disparity += np.pi * 2
        elif disparity > np.pi:
            disparity -= np.pi * 2
        elif disparity == -np.pi:
            disparity = np.pi
    else:
        disparity[disparity < -np.pi] += np.pi * 2
        disparity[disparity > np.pi] -= np.pi * 2
        disparity[disparity == -np.pi] = np.pi

    if not return_xyz:
        return disparity
    else:
        return disparity, xyz_all_l, xyz_all_r
