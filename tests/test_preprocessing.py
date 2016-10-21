from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from itertools import product

import numpy as np

from leelabtoolbox.preprocessing import transformers, pipeline

rng_state = np.random.RandomState(seed=0)

class MyTestCase(unittest.TestCase):
    def setUp(self):
        rng_state.seed(0)

    def test_random_sampling(self):

        numpatch_more = 10000
        numim_list = [1, 10]
        numpatches_list = [1, numpatch_more]
        patchsize_list = product(range(1, 4), range(1, 4))  # 16 combinations
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

            # call the transformer interface.
            result_4 = transformers.transformer_dict['sampling']({'type': 'random',
                                                                  'patchsize': patchsize,
                                                                  'random_numpatch': numpatches,
                                                                  'random_seed': 0,
                                                                  'fixed_locations': None,
                                                                  'verbose': False,
                                                                  'random_buff': buff,
                                                                  }).transform(image_list)
            self.assertTrue(np.array_equal(result, result_4))
            # call the transformer interface with returned locations.
            result_5 = transformers.transformer_dict['sampling']({'type': 'random',
                                                                  'patchsize': patchsize,
                                                                  'random_numpatch': numpatches,
                                                                  'random_seed': 1,
                                                                  'fixed_locations': location_list,
                                                                  'verbose': False,
                                                                  'random_buff': buff,
                                                                  }).transform(image_list)
            self.assertTrue(np.array_equal(result, result_5))

            if patchsize[0] == patchsize[1]:
                # call the transformer interface.
                result_4b = transformers.transformer_dict['sampling']({'type': 'random',
                                                                       'patchsize': patchsize[0],
                                                                       'random_numpatch': numpatches,
                                                                       'random_seed': 0,
                                                                       'fixed_locations': None,
                                                                       'verbose': False,
                                                                       'random_buff': buff,
                                                                       }).transform(image_list)
                self.assertTrue(np.array_equal(result, result_4b))
                # call the transformer interface with returned locations.
                result_5b = transformers.transformer_dict['sampling']({'type': 'random',
                                                                       'patchsize': patchsize[0],
                                                                       'random_numpatch': numpatches,
                                                                       'random_seed': 1,
                                                                       'fixed_locations': location_list,
                                                                       'verbose': False,
                                                                       'random_buff': buff,
                                                                       }).transform(image_list)
                self.assertTrue(np.array_equal(result, result_5b))

            if buff[0] == buff[1]:
                # call the transformer interface.
                result_4c = transformers.transformer_dict['sampling']({'type': 'random',
                                                                       'patchsize': patchsize,
                                                                       'random_numpatch': numpatches,
                                                                       'random_seed': 0,
                                                                       'fixed_locations': None,
                                                                       'verbose': False,
                                                                       'random_buff': buff[0],
                                                                       }).transform(image_list)
                self.assertTrue(np.array_equal(result, result_4c))
                # call the transformer interface with returned locations.
                result_5c = transformers.transformer_dict['sampling']({'type': 'random',
                                                                       'patchsize': patchsize,
                                                                       'random_numpatch': numpatches,
                                                                       'random_seed': 1,
                                                                       'fixed_locations': location_list,
                                                                       'verbose': False,
                                                                       'random_buff': buff[0],
                                                                       }).transform(image_list)
                self.assertTrue(np.array_equal(result, result_5c))

    def test_log(self):
        for _ in range(10):
            # generate a bunch of things in the range of 10 to 100, so that log()'s accuracy is enough.
            input_this = rng_state.rand(*rng_state.randint(1, 10, size=3)) * 90 + 10
            assert input_this.size > 0
            # generate a bias
            bias_this = np.clip(rng_state.randn() * 10, -5, 5)
            scale_factor_this = rng_state.rand() * 10
            ref_result = scale_factor_this * np.log(input_this + bias_this)
            returned_result = transformers.transformer_dict['logTransform']({'bias': bias_this,
                                                                             'scale_factor': scale_factor_this,
                                                                             'verbose': True
                                                                             }).transform(input_this)
            self.assertEqual(ref_result.shape, returned_result.shape)
            self.assertTrue(np.allclose(ref_result, returned_result, atol=1e-6))

    def test_gamma(self):
        for _ in range(10):
            input_this = rng_state.rand(*rng_state.randint(1, 10, size=3)) * 90 + 10
            assert input_this.size > 0
            # generate a bias
            gamma_this = rng_state.rand() * 10
            scale_factor_this = rng_state.rand() * 10
            ref_result = scale_factor_this * (input_this ** gamma_this)
            returned_result = transformers.transformer_dict['gammaTransform']({'gamma': gamma_this,
                                                                               'scale_factor': scale_factor_this,
                                                                               'verbose': True,
                                                                               }).transform(input_this)
            self.assertEqual(ref_result.shape, returned_result.shape)
            self.assertTrue(np.allclose(ref_result, returned_result, atol=1e-6))

    def test_mean(self):
        for _ in range(10):
            input_this = rng_state.randn(*rng_state.randint(1, 10, size=2))
            assert input_this.ndim == 2 and input_this.size > 0
            ref_result = input_this - input_this.mean(axis=1)[:, np.newaxis]
            returned_result = transformers.transformer_dict['removeDC']({}).transform(input_this)
            self.assertEqual(ref_result.shape, returned_result.shape)
            self.assertTrue(np.allclose(ref_result, returned_result, atol=1e-6))

    def test_unitvar_naive(self):
        for _ in range(10):
            # second dim must have more than 2 elements.
            input_this = rng_state.randn(rng_state.randint(1, 10), rng_state.randint(10, 50))
            assert input_this.ndim == 2 and input_this.size > 0 and input_this.shape[1] > 1
            ref_std = np.std(input_this, axis=1, keepdims=True)
            assert np.all(ref_std != 0)
            ref_result = input_this / ref_std
            # test most trivial case.
            returned_result = transformers.transformer_dict['unitVar']({
                'ddof': 0,
                'epsilon': 0,
                'epsilon_type': 'naive'
            }).transform(input_this)
            self.assertEqual(ref_result.shape, returned_result.shape)
            self.assertTrue(np.allclose(ref_result, returned_result, atol=1e-6))
            self.assertTrue(np.allclose(np.var(ref_result, axis=1), 1, atol=1e-6))

    def test_unitvar_naive_2(self):
        for _ in range(10):
            # second dim must have more than 2 elements.
            input_this = rng_state.randn(rng_state.randint(1, 10), rng_state.randint(10, 50))
            assert input_this.ndim == 2 and input_this.size > 0 and input_this.shape[1] > 1
            eps_this = rng_state.rand()
            ref_std = np.sqrt(np.var(input_this, axis=1, keepdims=True) + eps_this)
            assert np.all(ref_std != 0)
            ref_result = input_this / ref_std
            # test most trivial case.
            returned_result = transformers.transformer_dict['unitVar']({
                'ddof': 0,
                'epsilon': eps_this,
                'epsilon_type': 'naive'
            }).transform(input_this)
            self.assertEqual(ref_result.shape, returned_result.shape)
            self.assertTrue(np.allclose(ref_result, returned_result, atol=1e-6))

    def test_unitvar_quantile(self):
        for _ in range(10):
            # I use 101 elements to specify quantile very easily.
            input_this = rng_state.randn(101, rng_state.randint(10, 50))
            assert input_this.ndim == 2 and input_this.size > 0 and input_this.shape[1] > 1
            ref_var = np.var(input_this, axis=1, keepdims=True)
            quantile_to_use = rng_state.randint(0, 101)
            epsilon_actual = np.sort(ref_var.ravel())[quantile_to_use]
            ref_result = input_this / np.sqrt(ref_var + epsilon_actual)
            # test most trivial case.
            returned_result = transformers.transformer_dict['unitVar']({
                'ddof': 0,
                'epsilon': quantile_to_use / 100,
                'epsilon_type': 'quantile'
            }).transform(input_this)
            self.assertEqual(ref_result.shape, returned_result.shape)
            self.assertTrue(np.allclose(ref_result, returned_result, atol=1e-6))

    def test_pipeline(self):
        # this is just to increase coverage, so test won't be thorough (well you can't be thorough on anything anyways)
        # just test one case, as used in my ICA data generation code
        images_to_use = rng_state.rand(10, 1024, 1024)
        # # nothing at all.
        steps_naive = ['sampling', 'gammaTransform', 'flattening', 'removeDC', 'unitVar']
        pars_naive = {'sampling': {'type': 'random',
                                   'patchsize': 30, 'random_numpatch': 1000,
                                   'random_buff': 4,
                                   # just to somehow avoid some artefact on border. shouldn't matter.
                                   'random_seed': 0},
                      'unitVar': {
                          'epsilon': 0.1,  # 0.1 quantile (10 percentile).
                          'epsilon_type': 'quantile'  # follow Natural Image Statistics book.
                      }}
        pipeline_naive, realpars_naive, order_naive = pipeline.preprocessing_pipeline(steps_naive, pars_naive,
                                                                                      order=steps_naive)
        X_naive = pipeline_naive.transform(images_to_use)
        # ok, let's test if we can get these done manually.
        # first, do sampling
        pars_sampling = {
            'type': 'random',
            'patchsize': 30, 'random_numpatch': 1000,
            'random_buff': 4,
            # just to somehow avoid some artefact on border. shouldn't matter.
            'random_seed': 0,
            'fixed_locations': None,  # should be an iterable of len 1 or len of images, each
            # being a n_patch x 2 array telling the row and column of top left corner.
            'verbose': True
        }
        pars_gamma_transform = {'gamma': 0.5, 'scale_factor': 1.0, 'verbose': False}
        pars_flattening = {}
        pars_remove_dc = {}
        pars_unit_var = {
            'epsilon': 0.1,
            'ddof': 0,  # this is easier to understand.
            'epsilon_type': 'quantile'
        }
        # let's do it one by one.
        step_sampling = transformers.transformer_dict['sampling'](pars_sampling)
        step_gamma = transformers.transformer_dict['gammaTransform'](pars_gamma_transform)
        step_flattening = transformers.transformer_dict['flattening'](pars_flattening)
        step_remove_dc = transformers.transformer_dict['removeDC'](pars_remove_dc)
        step_unit_var = transformers.transformer_dict['unitVar'](pars_unit_var)

        X_ref = step_unit_var.transform(
            step_remove_dc.transform(
                step_flattening.transform(
                    step_gamma.transform(
                        step_sampling.transform(images_to_use)
                    )
                )
            )
        )

        self.assertTrue(np.array_equal(X_ref, X_naive))

if __name__ == '__main__':
    unittest.main()
