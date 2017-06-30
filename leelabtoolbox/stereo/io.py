"""helper functions to load various type of stereo datasets"""
from __future__ import absolute_import, division, print_function, unicode_literals
from struct import unpack
import numpy as np
from collections import OrderedDict
from scipy.io import loadmat

_brown_field_maping = {
    'range': 'range_m',
    'bearing': 'bearing_rad',
    'inclination': 'inclination_rad',
    'intensity': 'intensity_unk',
}


def read_brown_image_image_database(fname):
    """parse a single .bin file from brown image database (http://www.dam.brown.edu/ptg/brid/range/index.html).

    a rewrite of `readrange_float.m` in `brown_range/brown_range_public` of Lee Lab dataset server.

    below is copied from original dataset website


    The Brown Range Image Database contains 197 range images collected by Ann Lee and Jinggang Huang.
    Some preliminary analysis on these images were presented at CVPR, South Carolina, June 2000,
    and ICCV (2nd Int'l Workshop on Statistical and Computational Theories of Vision), Vancouver, July 2001.
    The conference papers can be downloaded from
    http://www.dam.brown.edu/ptg/brid/range/CVPR2000.ps.gz and http://www.cis.ohio-state.edu/~szhu/sctv01/Lee.html.

    Download the database here (250 MB)

    About the images:

    The images have been collected with a laser range-finder with a rotating mirror
    (3D imaging sensor LMS-Z210 by Riegl).
    Each image contains 444x1440 measurements with an angular separation of 0.18 deg.
    The field of view is thus 80 degrees vertically and 259 degrees horizontally.
    Each measurement is calculated from the time of flight of the laser beam.
    The operational range of the sensor is typically 2-200m.
    The laser wavelength of the range-finder is 0.9 mu m, which is in the near infra-red region.
    The data set consists of images which can be categorized as "forest", "residential", and "interior" scenes.

    All range image files are stored in a binary format and can be found in the directory binary/.
    Each image is represented in terms of 16 bit unsigned integer (uint16) elements stored by big-endian byte ordering.

    Every range image is stored as a structure with the following organization:


    Rows and Cols	- The number of rows and colums in the image is given by the first two uint16.
    Range	- The range image is stored by Rows x Cols uint16's in column order.
    Intensity	- The intensity image is stored by Rows x Cols uint16's in column order.
    Bearing	- The bearing coordinates is stored by Rows x Cols uint16's in column order.
    Inclination	- The inclination coordinates is stored by Rows x Cols uint16's in column order.
    The file src/readrange.m is a Matlab program for reading the images.

    About the variables in the .bin files:

    Range	- Distance (polar coordinates!) to object in units of 0.008 meters.
    A zero value means that the data point is missing; This will, for example, happen when an object is out of range.
    Intensity	- Reflectance image for laser range-finder (again, zero values where data is missing)
    Bearing	- Angles for vertical direction measured in units of 0.01 gons (400 gons for the full circle).
    The LRF starts scanning at about 0.89 rad or 51 deg, and ends at about 2.28 rad or 131 deg
    (0 is up, pi/2 is straight forward, and pi is down).
    Inclination	- Angles for horizontal direction (in 0.01 gons)
    """
    _header_format = '>2H'
    with open(fname, 'rb') as f:
        bytes_all = f.read()
    # then get height and width
    height, width = unpack(_header_format, bytes_all[:4])
    # then get all others
    # make sure it's correct length
    assert len(bytes_all) == 4 + 4 * (height * width * 2)
    # then read using numpy
    result = dict()

    key_and_process = OrderedDict()

    # unit is in 0.008 meters.
    key_and_process['range'] = lambda x: x.astype(np.float64) * 0.008
    key_and_process['intensity'] = lambda x: x
    # 0.01 gon to radian. % I think this should be multiplied by 2 to get in radian, as in Lee's version.
    key_and_process['bearing'] = lambda x: x.astype(np.float64) / 100 * np.pi * 2 / 400
    # 0.01 gon to radian.
    key_and_process['inclination'] = lambda x: x.astype(np.float64) / 100 * np.pi * 2 / 400
    for idx, (key, process_fn) in enumerate(key_and_process.items()):
        start_idx = 4 + idx * height * width * 2
        end_idx = 4 + (idx + 1) * height * width * 2
        bytes_this = bytes_all[start_idx:end_idx]
        # gives machine native representation.
        result[key] = np.frombuffer(bytes_this, dtype='>u2').astype(np.uint16).reshape((height, width), order='F')
        result[key] = process_fn(result[key])
        assert np.all(np.isfinite(result[key]))

    return result


def read_brown_image_image_database_lee(fname):
    """parse a single .mat file from leelab version of Brown database,

    available at `brown_range/brown_range_leelab` of Lee Lab dataset server.

    will give same output as in `read_brown_image_image_database`.
    """
    result_raw = loadmat(fname)

    result = dict()

    for new_f, old_f in _brown_field_maping.items():
        # unpack struct
        old_element = result_raw['r'][old_f][0,0]
        result[new_f] = old_element.astype(np.float64)
        assert np.all(np.isfinite(result[new_f]))

    return result
