from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np
import os.path

from leelabtoolbox.stereo import io, conversion

test_dir = os.path.split(__file__)[0]


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.rng_state = np.random.RandomState(seed=0)

    def test_io_browndataset(self):
        # check that results from the two are the same, as well as with raw lee version using loadmat.
        res1 = io.read_brown_image_image_database_lee(os.path.join(test_dir, 'stereo_ref', 'brown', 'V3_4.mat'))
        res2 = io.read_brown_image_image_database(os.path.join(test_dir, 'stereo_ref', 'brown', 'V3_4.bin'))
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

    def test_cart_and_sph_retina2(self):
        # it's unlikely that I will generate data with 0,0,0.
        numberOfCases = 20
        for i_case in range(numberOfCases):
            if i_case % 5 == 0:
                scalar_flag = True
            else:
                scalar_flag = False
            if not scalar_flag:
                # 1d to 3d
                # conversion to tuple is important.
                shape_this = tuple(self.rng_state.randint(low=1, high=200, size=self.rng_state.randint(low=1, high=4)))
            else:
                shape_this = ()
            xyz_this = self.rng_state.randn(3, *shape_this)
            if scalar_flag:
                assert xyz_this.shape == (3,)
            xyz_recompute = np.asarray(conversion.sph2cart(*conversion.cart2sph(*xyz_this, convention='retina2'),
                                                           convention='retina2'))
            self.assertEqual(xyz_this.shape, xyz_recompute.shape)
            self.assertTrue(np.allclose(xyz_this, xyz_recompute, atol=1e-8))

            dll_this = np.empty((3,) + shape_this, dtype=np.float64)
            # d
            dll_this[0] = self.rng_state.rand(*shape_this)
            # latitude
            dll_this[1] = self.rng_state.rand(*shape_this) * np.pi - np.pi / 2
            # longitude
            dll_this[2] = self.rng_state.rand(*shape_this) * 2 * np.pi - np.pi
            if scalar_flag:
                assert dll_this.shape == (3,)

            dll_recompute = np.asarray(conversion.cart2sph(*conversion.sph2cart(*dll_this, convention='retina2'),
                                                           convention='retina2'))
            self.assertEqual(dll_this.shape, dll_recompute.shape)
            self.assertTrue(np.allclose(dll_this, dll_recompute, atol=1e-8))

    def test_car_and_sph_standard(self):
        numberOfCases = 20
        for i_case in range(numberOfCases):
            if i_case % 5 == 0:
                scalar_flag = True
            else:
                scalar_flag = False
            if not scalar_flag:
                # 1d to 3d
                # conversion to tuple is important.
                shape_this = tuple(self.rng_state.randint(low=1, high=200, size=self.rng_state.randint(low=1, high=4)))
            else:
                shape_this = ()
            xyz_this = self.rng_state.randn(3, *shape_this)
            if scalar_flag:
                assert xyz_this.shape == (3,)
            xyz_recompute = np.asarray(conversion.sph2cart(*conversion.cart2sph(*xyz_this, convention='standard'),
                                                           convention='standard'))
            self.assertEqual(xyz_this.shape, xyz_recompute.shape)
            self.assertTrue(np.allclose(xyz_this, xyz_recompute, atol=1e-8))

            dll_this = np.empty((3,) + shape_this, dtype=np.float64)
            # d
            dll_this[0] = self.rng_state.rand(*shape_this)
            # latitude
            dll_this[1] = self.rng_state.rand(*shape_this) * np.pi - np.pi / 2
            # longitude
            dll_this[2] = self.rng_state.rand(*shape_this) * 2 * np.pi
            if scalar_flag:
                assert dll_this.shape == (3,)

            dll_recompute = np.asarray(conversion.cart2sph(*conversion.sph2cart(*dll_this, convention='standard'),
                                                           convention='standard'))
            self.assertEqual(dll_this.shape, dll_recompute.shape)
            self.assertTrue(np.allclose(dll_this, dll_recompute, atol=1e-8))


if __name__ == '__main__':
    unittest.main()
