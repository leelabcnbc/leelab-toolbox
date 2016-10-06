from __future__ import absolute_import, division, print_function, unicode_literals
import numbers
import numpy as np


def normalize_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return a new randomState with seed None, so it's impossible to reproduce the result.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.

    this function is modified from `check_random_state` of sklearn.

    """
    if seed is None:
        return np.random.RandomState(seed=None)
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState instance'.format(seed))


def make_2d_array(old_array):
    # make flat, if not ndarray or at least 3d.
    old_array = np.asarray(old_array)
    assert old_array.size > 0, 'degenerate input!'
    if old_array.ndim <= 2:
        result = np.atleast_2d(old_array)
    else:
        result = old_array.reshape(old_array.shape[0], -1)
    return result
