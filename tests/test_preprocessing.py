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
                # check that every possible one is covered, no more, no less.
                location_this = location_list[0]
                set_locations_actual = set([tuple(x) for x in location_this])
                correct_h = np.arange(buff[0], height_this - buff[0] - patchsize[0] + 1)
                correct_w = np.arange(buff[1], width_this - buff[1] - patchsize[1] + 1)
                set_locations_theory = set(product(correct_h, correct_w))
                self.assertEqual(set_locations_actual, set_locations_theory)

            # check that you can recover the images by passind the locations, using another seed, which is in any case
            # not used at all.
            result_2, location_list_2 = transformers.sampling_random(image_list, patchsize, numpatches, buff=buff,
                                                                     seed=1, fixed_locations=location_list,
                                                                     return_locations=True)

            self.assertTrue(np.array_equal(result, result_2))
            self.assertEqual(len(location_list), len(location_list_2))
            for idx in range(len(location_list)):
                self.assertTrue(np.array_equal(location_list[idx],
                                               location_list_2[idx]))

            # check that you can recover the images using location_list_2 manually.
            result_3 = []
            for idx, location_this in enumerate(location_list):
                image_this = image_list[idx]
                im_h, im_w = image_this.shape
                for (h, w) in location_this:
                    assert buff[0] <= h <= (im_h - buff[0] - patchsize[0])
                    assert buff[1] <= w <= (im_w - buff[1] - patchsize[1])
                    result_3.append(image_this[h:h + patchsize[0], w:w + patchsize[1]])
            result_3 = np.asarray(result_3)
            self.assertTrue(np.array_equal(result, result_3))


if __name__ == '__main__':
    unittest.main()
