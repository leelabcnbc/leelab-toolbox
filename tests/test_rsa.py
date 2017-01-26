from __future__ import absolute_import, division, print_function
import unittest
from leelabtoolbox.rsa import (compute_rdm_bounds, compute_rdm)
import numpy as np
import h5py
import os.path
from scipy.spatial.distance import squareform

test_dir = os.path.split(__file__)[0]


class MyTestCase(unittest.TestCase):
    def test_rsa_bounds_spearman(self):
        with h5py.File(os.path.join(test_dir, 'rsa_ref', 'rsa_ref.hdf5'), 'r') as f:
            rdm_stack_all = f['rsa_bounds/rdm_stack_all'][...]  # don't need transpose, as it's symmetric.
            result_array = f['rsa_bounds/result_array'][...].T
        for idx, rdm_stack in enumerate(rdm_stack_all):
            rdm_vector_list = []
            for rdm in rdm_stack:
                rdm_vector_list.append(squareform(rdm))
            rdm_vector_list = np.array(rdm_vector_list)
            lower_bound, upper_bound = compute_rdm_bounds(rdm_vector_list, similarity_type='spearman', legacy=True)
            lower_bound_ref = result_array[idx, 1]
            upper_bound_ref = result_array[idx, 0]
            self.assertEqual(upper_bound.shape, upper_bound_ref.shape)
            self.assertEqual(lower_bound.shape, lower_bound_ref.shape)
            self.assertTrue(np.allclose(lower_bound, lower_bound_ref))
            self.assertTrue(np.allclose(upper_bound, upper_bound_ref))

    def test_rsa_computation(self):
        with h5py.File(os.path.join(test_dir, 'rsa_ref', 'rsa_ref.hdf5'), 'r') as f:
            rdm_stack_all = f['rsa_bounds/rdm_stack_all'][...]  # don't need transpose, as it's symmetric.
            feature_stack_all = f['rsa_bounds/feature_matrix_all'][...]
        # ok. let's compute the rdm in two ways
        assert len(rdm_stack_all) == len(feature_stack_all)
        for idx, feature_this in enumerate(feature_stack_all):
            # compute all rdms
            rdm_list_this_1 = np.asarray([squareform(compute_rdm(X.T, method='scipy')) for X in feature_this])
            rdm_list_this_2 = np.asarray([squareform(compute_rdm(X.T, method='sklearn')) for X in feature_this])
            self.assertTrue(np.allclose(rdm_list_this_1, rdm_stack_all[idx]))
            self.assertTrue(np.allclose(rdm_list_this_2, rdm_stack_all[idx]))
            self.assertEqual(rdm_list_this_1.shape, rdm_stack_all[idx].shape)
            self.assertEqual(rdm_list_this_2.shape, rdm_stack_all[idx].shape)
