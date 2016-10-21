from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np

from leelabtoolbox import util

rng_state = np.random.RandomState(seed=0)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        rng_state.seed(0)

    def test_make_2d_array(self):
        # try 0d, 1d, and 2d
        # 0d, scalar.
        zerod_test_input = rng_state.randn()
        zerod_test_ref_result = np.array([[zerod_test_input]])
        zerod_test_result = util.make_2d_array(zerod_test_input)
        self.assertTrue(np.array_equal(zerod_test_result, zerod_test_ref_result))

        oned_test_input = rng_state.randn(10)
        oned_test_ref_result = oned_test_input[np.newaxis, :].copy()
        oned_test_result = util.make_2d_array(oned_test_ref_result)
        self.assertTrue(np.array_equal(oned_test_result, oned_test_ref_result))

        twod_test_input = rng_state.randn(10, 10)
        twod_test_ref_result = twod_test_input.copy()
        oned_test_result = util.make_2d_array(twod_test_input)
        self.assertTrue(np.array_equal(oned_test_result, twod_test_ref_result))

        threed_test_input = rng_state.randn(10, 10, 10)
        threed_test_ref_result = threed_test_input.reshape(10, -1, order='C').copy()
        threed_test_result = util.make_2d_array(threed_test_input)
        self.assertTrue(np.array_equal(threed_test_ref_result, threed_test_result))

    def test_display_network(self):
        # I do this primarily to increase coverage
        random_image = rng_state.rand(100, 100) * 2 - 1
        random_image[0, 0] = 1  # to avoid scaling.
        self.assertTrue(np.all(random_image >= -1) and np.all(random_image <= 1))
        padding_this = rng_state.randint(1, 10)
        ref_result = np.ones((padding_this * 2 + 100, padding_this * 2 + 100), dtype=np.float64)
        ref_result[(ref_result.shape[0] // 2 - 50):(ref_result.shape[0] // 2 + 50),
        (ref_result.shape[1] // 2 - 50):(ref_result.shape[1] // 2 + 50)] = random_image
        test_result = util.display_network(random_image.ravel()[np.newaxis, :], padding=padding_this)
        self.assertTrue(np.array_equal(test_result, ref_result))
