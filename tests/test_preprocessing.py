from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from itertools import product

import numpy as np

from leelabtoolbox.preprocessing import transformers

rng_state = np.random.RandomState(seed=0)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        rng_state.seed(0)

    def test_random_sampling(self):
        numpatch_more = 10000
        numim_list = [1, 10]
        numpatches_list = [1, numpatch_more]
        patchsize_list = product(range(1, 6), range(1, 6))  # 25 combinations
        buff_list = product(range(3), range(3))  # 9 combinations
        for trial_idx, (numim, numpatches, patchsize, buff) in enumerate(
                product(numim_list, numpatches_list, patchsize_list, buff_list)):
            # subTest is not supported in 2.7
            # with self.subTest(numim=numim, numpatches=numpatches, patchsize=patchsize, buff=buff):
                # at least create images of size bigger than (buff[0]*2+patchsize[0]) by (buff[1]*2+patchsize[1])
                # add add at most 5
                min_height = buff[0] * 2 + patchsize[0]
                min_width = buff[1] * 2 + patchsize[1]
                image_list = []
                for _ in range(numim):
                    height_this = rng_state.randint(min_height, min_height + 5 + 1)
                    width_this = rng_state.randint(min_width, min_width + 5 + 1)
                    image_list.append(rng_state.randn(height_this, width_this))
                result, location_list = transformers.sampling_random(image_list, patchsize, numpatches, buff=buff,
                                                                     seed=0, return_locations=True)
                if numpatches == numpatch_more and numim == 1:
                    # check that every possible one is covered.
                    unique_h = np.unique(location_list[0][:, 0])
                    unique_w = np.unique(location_list[0][:, 1])
                    correct_h = np.arange(buff[0], height_this - buff[0] - patchsize[0] + 1)
                    correct_w = np.arange(buff[1], width_this - buff[1] - patchsize[1] + 1)
                    self.assertTrue(np.array_equal(unique_h, correct_h))
                    self.assertTrue(np.array_equal(unique_w, correct_w))


if __name__ == '__main__':
    unittest.main()
