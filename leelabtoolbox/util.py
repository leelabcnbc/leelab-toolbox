from __future__ import absolute_import, division, print_function, unicode_literals
import numbers
from itertools import product
import numpy as np
from sklearn.preprocessing import maxabs_scale


def normalize_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return a new randomState with seed None, so it's impossible to reproduce the result.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.

    this function is modified from `check_random_state` of sklearn.

    """
    if seed is None or isinstance(seed, numbers.Integral):
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


def display_network(W, n_col=None, n_row=None, transpose=False, padding=1, image_shape=None):
    """visualizing
    :param W:
    :param transpose:
    :return:
    """
    # scale each one to [-1, 1]
    assert W.ndim == 2
    # TODO: add other normalization behaviour
    W = maxabs_scale(W, axis=1)
    n_basis, n_pixel = W.shape
    if image_shape is None:
        image_shape = int(np.sqrt(n_pixel)), int(np.sqrt(n_pixel))
    assert image_shape[0] * image_shape[1] == n_pixel
    if n_col is None and n_row is None:
        n_col = int(np.ceil(np.sqrt(n_basis)))
        n_row = int(np.ceil(float(n_basis) / n_col))
    cell_height = image_shape[0] + 2 * padding
    cell_width = image_shape[1] + 2 * padding
    total_image = np.ones(shape=(n_row * cell_height, n_col * cell_width),
                          dtype=np.float64)

    for idx, (row_idx, col_idx) in enumerate(product(range(n_row), range(n_col))):
        if idx >= n_basis:
            break

        position_to_plot = (slice(row_idx * cell_height + padding, row_idx * cell_height + padding + image_shape[0]),
                            slice(col_idx * cell_width + padding, col_idx * cell_width + padding + image_shape[1]))
        cell_this = W[idx].reshape(image_shape)
        if transpose:
            cell_this = cell_this.T
        total_image[position_to_plot] = cell_this

    return total_image
