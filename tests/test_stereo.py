from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np
import os.path

from leelabtoolbox.stereo import io

test_dir = os.path.split(__file__)[0]


class MyTestCase(unittest.TestCase):
    def test_io_browndataset(self):
        # check that results from the two are the same, as well as with raw lee version using loadmat.
        res1 = io.read_brown_image_image_database_lee(os.path.join(test_dir, 'stereo_ref', 'brown', 'V1_4.mat'))
        res2 = io.read_brown_image_image_database(os.path.join(test_dir, 'stereo_ref', 'brown', 'V1_4.bin'))
        self.assertEqual(set(res1.keys()), set(res2.keys()))

        _error_standards = {
            'range': 0.01,
            'bearing': 1e-3,
            'inclination': 1e-3,
            'intensity': 1e-6,
        }

        for key in res1:
            v1 = res1[key]
            v2 = res2[key]
            self.assertTrue(v1.shape == v2.shape)
            self.assertTrue(abs(v1 - v2).max() < _error_standards[key])


if __name__ == '__main__':
    unittest.main()
