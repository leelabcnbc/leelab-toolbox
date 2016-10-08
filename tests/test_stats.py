from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from leelabtoolbox import stats
from scipy.spatial import distance
import numpy as np
from itertools import product

rng_state = np.random.RandomState(seed=0)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        rng_state.seed(0)

    def test_sparsity_measure_rolls(self):
        for _ in range(10):
            # first, set up a nonnegative vector.
            #vector_length_list = [1, rng_state.randint(2, 100)]
            vector_length_list = [rng_state.randint(2, 100)]
            zero_response_list = [True, False]
            for vec_length, zero_response in product(vector_length_list, zero_response_list):
                #print('vec of length {}'.format(vec_length))
                vector_this = rng_state.randn(vec_length)
                if zero_response:
                    vector_this[...] = 0
                vector_this_nn = np.clip(vector_this, 0, np.inf)
                if np.all(vector_this_nn == 0):
                    vector_this_nn[:] = 1.0  # this is how we deal with all zero vector.
                if np.all(vector_this == 0):
                    vector_this_ = np.ones_like(vector_this)  # this is how we deal with all zero vector.
                else:
                    vector_this_ = vector_this.copy()
                self.assertTrue(np.any(vector_this_nn > 0) and np.all(vector_this_nn >= 0))

                for normalize_par, clip_par in product([True, False], [True, False]):
                    if (not normalize_par) and (not clip_par):
                        sparsity_ref = 1 - (1 - distance.cosine(vector_this_, np.ones_like(vector_this_))) ** 2
                    elif (not normalize_par) and clip_par:
                        sparsity_ref = 1 - (1 - distance.cosine(vector_this_nn, np.ones_like(vector_this_))) ** 2
                    elif normalize_par and (not clip_par):
                        sparsity_ref = 1 - (1 - distance.cosine(vector_this_, np.ones_like(vector_this_))) ** 2
                        sparsity_ref /= (1 - 1 / vec_length)
                    else:
                        assert normalize_par and clip_par
                        sparsity_ref = 1 - (1 - distance.cosine(vector_this_nn, np.ones_like(vector_this_))) ** 2
                        sparsity_ref /= (1 - 1 / vec_length)
                    sparsity_computed = stats.sparsity_measure_rolls(vector_this,
                                                                     normalize=normalize_par, clip=clip_par)
                    self.assertAlmostEqual(sparsity_computed, sparsity_ref)


if __name__ == '__main__':
    unittest.main()
