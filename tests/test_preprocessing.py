from __future__ import absolute_import, division, print_function, unicode_literals

import os.path
import unittest
from itertools import product

import h5py
import numpy as np

from leelabtoolbox.preprocessing import transformers, pipeline

test_dir = os.path.split(__file__)[0]

rng_state = np.random.RandomState(seed=0)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        rng_state.seed(0)

    def gen_length(self, odd, base=0, min=10, max=20):
        this_len = rng_state.randint(min, max + 1)
        this_len += base
        if this_len % 2 != int(odd):
            this_len += 1
        return this_len

    def generate_actual_output(self, h, w, shape, bg_color, images, pos_list):
        # print(h, w, shape, images.shape)
        assert pos_list.shape == (len(images), 2)
        assert bg_color.shape == shape
        images_result = np.empty((len(images), h, w) + shape, dtype=np.float64)
        images_result[...] = bg_color

        for image_big, image_ref, (j_h, j_w) in zip(images_result, images, pos_list):
            h_img, w_img = image_ref.shape[:2]
            assert j_h >= 0 and j_w >= 0 and j_h + h_img <= h and j_w + w_img <= w
            image_big[j_h:j_h + h_img, j_w:j_w + w_img] = image_ref
        return images_result

    def get_pos_early_vision_tbx(self, center, img_h, img_w, jitter):
        # this is original equation for computing location.
        # we only need the left end point.
        # row_index = np.floor(np.arange(crows - rowsthis / 2.0 + jitterthisr, crows + rowsthis / 2.0 + jitterthisr))
        # col_index = np.floor(np.arange(ccols - colsthis / 2.0 + jitterthisc, ccols + colsthis / 2.0 + jitterthisc))

        row_first = np.floor(center[0] - img_h / 2.0 + jitter[0])
        col_first = np.floor(center[1] - img_w / 2.0 + jitter[1])

        return row_first, col_first

    def get_jitter_early_vision_tbx(self, center, img_h, img_w, pos):
        # this is original equation for computing location.
        # we only need the left end point.
        # row_index = np.floor(np.arange(crows - rowsthis / 2.0 + jitterthisr, crows + rowsthis / 2.0 + jitterthisr))
        # col_index = np.floor(np.arange(ccols - colsthis / 2.0 + jitterthisc, ccols + colsthis / 2.0 + jitterthisc))

        row_first = pos[0] - np.floor(center[0] - img_h / 2.0)
        col_first = pos[1] - np.floor(center[1] - img_w / 2.0)

        return row_first, col_first

    def test_put_in_canvas(self):
        # test jitter, no jitter,
        # for jitter one, test focus on location coverage.
        # for non jitter one, test focus on location precision.
        # test 5 different shapes. (), (1,), (3,), (1,3), (3,1)
        # test all combinations of all canvas size and image size combination.
        # (odd, even) x (odd, even), for both height and width
        # here, the convention should follow the old early_vision_toolbox's convention.
        # each time test 1 or 200 image.
        # the 200 one is mainly for coverage purpose for jitter.
        # the 1 one is for avoiding some potential bugs related to singleton dimensions.
        image_count_list = [1, 200]
        jitter_or_not = [True, False]
        # different types of jitter. they are small enough so that they with 200 images, they should be exhaustively
        # covered.
        jitter_pixel_list = [(1, 1), (1, 2), (2, 1), 1, (2, 0), (1, 0), 0]
        shape_list = [(), (1,), (3,), (1, 3), (3, 1)]
        with_center_list = [True, False]
        canvas_odd_h_list = [True, False]
        canvas_odd_w_list = [True, False]
        image_odd_h_list = [True, False]
        image_odd_w_list = [True, False]

        flag_non_standard_center = False

        for trial_idx, (num_im, jitter,
                        jitter_pixel,
                        pixel_shape, with_center,
                        canvas_odd_h, canvas_odd_w,
                        image_odd_h, image_odd_w) in enumerate(product(image_count_list, jitter_or_not,
                                                                       jitter_pixel_list,
                                                                       shape_list, with_center_list,
                                                                       canvas_odd_h_list, canvas_odd_w_list,
                                                                       image_odd_h_list, image_odd_w_list)):
            # remove some useless cases, to speed up.
            if not jitter and jitter_pixel != 0:
                continue
            if jitter and jitter_pixel == 0:
                continue
            if jitter and pixel_shape not in {(), (1,)}:
                # that one is just to test shape, don't need to mingle together.
                continue
            # print(num_im, jitter, jitter_pixel, pixel_shape, with_center,
            #       canvas_odd_h, canvas_odd_w,
            #       image_odd_h, image_odd_w)
            img_h, img_w = self.gen_length(image_odd_h), self.gen_length(image_odd_w)
            canvas_h, canvas_w = self.gen_length(canvas_odd_h, base=img_h), self.gen_length(canvas_odd_w, base=img_w)
            # print(img_h, img_w, canvas_h, canvas_w)
            # ok. let's generate image
            image_original = rng_state.randn(num_im, img_h, img_w, *pixel_shape)
            canvas_color = rng_state.randn(*pixel_shape)
            # then call it.

            if with_center:
                center_loc = canvas_h / 2.0 + rng_state.randint(-2, 3), canvas_w / 2.0 + rng_state.randint(-2, 3)
                # this must happen at least once
                if center_loc != (canvas_h / 2.0, canvas_w / 2.0):
                    flag_non_standard_center = True
                center_loc_numerical = center_loc
            else:
                center_loc = None
                center_loc_numerical = (canvas_h / 2.0, canvas_w / 2.0)

            returned_result, jitter_list, pos_list = transformers.transformer_dict['putInCanvas']({
                'canvas_size': (canvas_h, canvas_w),
                'canvas_color': canvas_color,
                'center_loc': center_loc,
                'jitter': jitter,
                'jitter_maxpixel': jitter_pixel,
                'jitter_seed': 0,
                'external_jitter_list': None,
                'return_jitter_list': True,
                'return_pos_list': True,
            }).transform(image_original.copy())

            returned_result_2 = transformers.transformer_dict['putInCanvas']({
                'canvas_size': (canvas_h, canvas_w),
                'canvas_color': canvas_color,
                'center_loc': center_loc,
                'jitter': jitter,
                'jitter_maxpixel': jitter_pixel,
                'jitter_seed': 0,
                'external_jitter_list': None,
                'return_jitter_list': False,
                'return_pos_list': False,
            }).transform(image_original.copy())

            returned_result_3 = transformers.transformer_dict['putInCanvas']({
                'canvas_size': (canvas_h, canvas_w),
                'canvas_color': canvas_color,
                'center_loc': center_loc,
                'jitter': jitter,
                'jitter_maxpixel': jitter_pixel,
                'jitter_seed': None,
                'external_jitter_list': jitter_list,
                'return_jitter_list': False,
                'return_pos_list': False,
            }).transform(image_original.copy())

            # three results are the same
            self.assertTrue(np.array_equal(returned_result, returned_result_2))
            self.assertTrue(np.array_equal(returned_result, returned_result_3))

            # ok. let's check.
            self.assertEqual(returned_result.shape, (num_im, canvas_h, canvas_w,) + pixel_shape)
            self.assertEqual(jitter_list.shape, (num_im, 2))
            returned_result_ref = self.generate_actual_output(canvas_h, canvas_w, pixel_shape, np.asarray(canvas_color),
                                                              image_original.copy(), pos_list)
            # this is already a proof that pos_list is correct.
            self.assertTrue(np.array_equal(returned_result, returned_result_ref))
            # then let's based on pos_list to get jitter list
            pos_list_debug = np.array(
                [self.get_pos_early_vision_tbx(center_loc_numerical, img_h, img_w, jitter_this) for jitter_this in
                 jitter_list])
            if not np.array_equal(pos_list_debug, pos_list):
                print(pos_list, pos_list_debug)
            self.assertTrue(np.array_equal(pos_list, pos_list_debug))
            jitter_list_debug = np.array(
                [self.get_jitter_early_vision_tbx(center_loc_numerical, img_h, img_w, pos_this) for pos_this in
                 pos_list])
            self.assertTrue(np.array_equal(jitter_list, jitter_list_debug))

            # check jitter is in correct range.
            jitter_range = np.broadcast_to(jitter_pixel, (2,))
            self.assertTrue(np.all(abs(jitter_list) <= jitter_range))

            # then for n_im = 200, check coverage of jitter
            if num_im == 200:
                self.assertTrue(
                    np.array_equal(np.unique(jitter_list[:, 0]), np.arange(-jitter_range[0], jitter_range[0] + 1)))
                self.assertTrue(
                    np.array_equal(np.unique(jitter_list[:, 1]), np.arange(-jitter_range[1], jitter_range[1] + 1)))

            # check that for matching parity case, it's exact

            if not jitter and not with_center and canvas_odd_h == image_odd_h:
                self.assertTrue(np.all(pos_list[:, 0] == pos_list[0, 0]))
                self.assertEqual(2 * pos_list[0, 0], canvas_h - img_h)

            if not jitter and not with_center and canvas_odd_w == image_odd_w:
                # print(pos_list[0, 1], canvas_w, img_w, canvas_odd_w, image_odd_w)
                self.assertTrue(np.all(pos_list[:, 1] == pos_list[0, 1]))
                self.assertEqual(2 * pos_list[0, 1], canvas_w - img_w)
        self.assertTrue(flag_non_standard_center)

    def test_aperture(self):
        gaussian_width_list = [0.0, 2]
        image_count_list = [1, 10]
        shape_list = [(), (1,), (3,), (1, 3), (3, 1)]
        shift_list = [(0, 0), (0, 3), (3, 0), (-3, 0), (0, -3), (3, 3), (-3, -3), (3, -3), (-3, 3)]
        legacy_or_not = [True, False]
        for trial_idx, (num_im,
                        pixel_shape, shift,
                        gaussian_width, legacy) in enumerate(product(image_count_list,
                                                                     shape_list, shift_list,
                                                                     gaussian_width_list,
                                                                     legacy_or_not)):
            # just make sure that I can run them. the actual test is done in examples.
            img_h = rng_state.randint(20, 30)
            img_w = rng_state.randint(20, 30)
            aperture_size = rng_state.randint(5, 10)
            image_original = rng_state.randn(num_im, img_h, img_w, *pixel_shape)

            padding = rng_state.randint(1, 5)

            image_original_inner = image_original[:, padding:-padding, padding:-padding].copy()
            canvas_color = rng_state.randn(*pixel_shape)

            returned_result = transformers.transformer_dict['aperture']({
                'size': aperture_size,
                'gaussian_width': gaussian_width,
                'shift': shift,
                'background_color': canvas_color,  # gray background by default.
                'legacy': legacy,
            }).transform(image_original.copy())
            self.assertEqual(returned_result.shape, image_original.shape)

            returned_result_inner = transformers.transformer_dict['aperture']({
                'size': aperture_size,
                'gaussian_width': gaussian_width,
                'shift': shift,
                'background_color': canvas_color,  # gray background by default.
                'legacy': legacy,
            }).transform(image_original_inner.copy())
            returned_result_cropped = returned_result[:, padding:-padding, padding:-padding]
            self.assertEqual(returned_result_inner.shape, returned_result_cropped.shape)
            if gaussian_width != 0 and shift == (0, 0):
                if not legacy:
                    self.assertTrue(np.allclose(returned_result_cropped, returned_result_inner, atol=1e-6))
                else:
                    self.assertFalse(np.allclose(returned_result_cropped, returned_result_inner, atol=1e-6))

    def test_random_sampling(self):

        numpatch_more = 10000
        numim_list = [1, 10]
        numpatches_list = [1, numpatch_more]
        patchsize_list = product(range(1, 4), range(1, 4))  # 16 combinations
        buff_list = product(range(3), range(3))  # 9 combinations
        nan_level_list = [None, 'some']
        for trial_idx, (numim, numpatches, patchsize, buff, nan_level_this) in enumerate(
                product(numim_list, numpatches_list, patchsize_list, buff_list, nan_level_list)):
            print(numim, numpatches, patchsize, buff, nan_level_this)
            # subTest is not supported in 2.7
            # with self.subTest(numim=numim, numpatches=numpatches, patchsize=patchsize, buff=buff):
            # at least create images of size bigger than (buff[0]*2+patchsize[0]) by (buff[1]*2+patchsize[1])
            # add add at most 5
            min_height = buff[0] * 2 + patchsize[0]
            min_width = buff[1] * 2 + patchsize[1]

            if nan_level_this is None:
                nan_level = None
            else:
                assert nan_level_this == 'some'
                nan_level = rng_state.rand()

            image_list = []

            for _ in range(numim):
                height_this = rng_state.randint(min_height, min_height + 5 + 1)
                width_this = rng_state.randint(min_width, min_width + 5 + 1)
                image_list.append(rng_state.randn(height_this, width_this))
            result, location_list = transformers.sampling_random(image_list, patchsize, numpatches, buff=buff,
                                                                 seed=0, return_locations=True, nan_level=nan_level,
                                                                 no_nan_loc=None)
            if nan_level is not None:
                # check that when changing no_nan_loc, I don't get any difference.
                # this is a pretty weak check.
                result_debug, location_list_debug = transformers.sampling_random(image_list, patchsize, numpatches,
                                                                                 buff=buff,
                                                                                 seed=0, return_locations=True,
                                                                                 nan_level=nan_level,
                                                                                 no_nan_loc=(slice(None), slice(None)))
                self.assertTrue(np.array_equal(result, result_debug))
                self.assertEqual(len(location_list), len(location_list_debug))
                for idx in range(len(location_list)):
                    self.assertTrue(np.array_equal(location_list[idx],
                                                   location_list_debug[idx]))

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
                                                                     return_locations=True, nan_level=nan_level,
                                                                     no_nan_loc=None)

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
                                                                  'nan_level': nan_level,
                                                                  'no_nan_loc': None,
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
                                                                  'nan_level': nan_level,
                                                                  'no_nan_loc': None,
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
                                                                       'nan_level': nan_level,
                                                                       'no_nan_loc': None,
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
                                                                       'nan_level': nan_level,
                                                                       'no_nan_loc': None,
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
                                                                       'nan_level': nan_level,
                                                                       'no_nan_loc': None,
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
                                                                       'nan_level': nan_level,
                                                                       'no_nan_loc': None,
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
            'verbose': True,
            'nan_level': None,
            'no_nan_loc': None,
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

    def test_whiten_olsh_lee_inner(self):
        # first, load files.
        with h5py.File(os.path.join(test_dir, 'preprocessing_ref', 'one_over_f_whitening_ref.hdf5'), 'r') as f:
            original_images = f['original_images'][...].transpose((0, 2, 1))
            new_images_default = f['new_images_default'][...].transpose((0, 2, 1))
            new_images_change_f0 = f['new_images_change_f0'][...].transpose((0, 2, 1))
            new_images_no_cutoff = f['new_images_no_cutoff'][...].transpose((0, 2, 1))
            new_images_change_crop = f['new_images_change_crop'][...].transpose((0, 2, 1))
            new_images_change_crop_pure = f['new_images_change_crop_pure'][...].transpose((0, 2, 1))

        # first, test default argument
        steps_naive = ['oneOverFWhitening']
        pars_naive = {'oneOverFWhitening':
                          {}
                      }

        pars_change_f0 = {'oneOverFWhitening':
                              {'f_0': 40,  # cut off frequency, in cycle / image. 0.4*mean(H, W) by default
                               }
                          }

        pars_no_cutoff = {'oneOverFWhitening':
            {
                'cutoff': False,  # whether do cutoff frequency or not.
            }
        }

        pars_change_crop = {'oneOverFWhitening':
            {
                'central_clip': (64, 128),
            }
        }

        pars_change_crop_pure = {'oneOverFWhitening':
            {
                'central_clip': (64, 128),
                # clip the central central_clip[0] x central_clip[1] part in the frequency
                # domain. by default, don't do anything.
                'no_filter': True,  # useful when only want to do central_clip,
            }
        }

        pipeline_default = pipeline.preprocessing_pipeline(steps_naive, pars_naive, order=steps_naive)[0]
        pipeline_change_f0 = pipeline.preprocessing_pipeline(steps_naive, pars_change_f0, order=steps_naive)[0]
        pipeline_cutoff = pipeline.preprocessing_pipeline(steps_naive, pars_no_cutoff, order=steps_naive)[0]
        pipeline_change_crop = pipeline.preprocessing_pipeline(steps_naive, pars_change_crop, order=steps_naive)[0]
        pipeline_change_crop_pure = pipeline.preprocessing_pipeline(steps_naive, pars_change_crop_pure,
                                                                    order=steps_naive)[0]

        def check_shape_and_close(X1, X2, no_filter=False):
            if not no_filter:
                # mean very close to zero
                self.assertTrue(abs(X2.mean()) < 1e-6)
            self.assertEqual(X1.shape, X2.shape)
            print(abs(X1 - X2).max())
            self.assertTrue(np.allclose(X1, X2))

        check_shape_and_close(new_images_default, pipeline_default.transform(original_images))
        check_shape_and_close(new_images_change_f0, pipeline_change_f0.transform(original_images))
        check_shape_and_close(new_images_no_cutoff, pipeline_cutoff.transform(original_images))
        check_shape_and_close(new_images_change_crop, pipeline_change_crop.transform(original_images))
        check_shape_and_close(new_images_change_crop_pure, pipeline_change_crop_pure.transform(original_images), True)

    def test_whiten_olsh_lee_inner_multi(self):
        # first, load files.
        with h5py.File(os.path.join(test_dir, 'preprocessing_ref', 'one_over_f_whitening_ref.hdf5'), 'r') as f:
            original_images = f['original_images'][...].transpose((0, 2, 1))
            new_images_default = f['new_images_default'][...].transpose((0, 2, 1))
            new_images_change_f0 = f['new_images_change_f0'][...].transpose((0, 2, 1))
            new_images_no_cutoff = f['new_images_no_cutoff'][...].transpose((0, 2, 1))
            new_images_change_crop = f['new_images_change_crop'][...].transpose((0, 2, 1))
            new_images_change_crop_pure = f['new_images_change_crop_pure'][...].transpose((0, 2, 1))

        # first, test default argument
        steps_naive = ['oneOverFWhitening']
        pars_naive = {'oneOverFWhitening':
            {
                'n_jobs': -1,
            }
        }

        pars_change_f0 = {'oneOverFWhitening':
                              {'f_0': 40,  # cut off frequency, in cycle / image. 0.4*mean(H, W) by default
                               'n_jobs': -1,
                               }
                          }

        pars_no_cutoff = {'oneOverFWhitening':
            {
                'cutoff': False,  # whether do cutoff frequency or not.
                'n_jobs': -1,
            }
        }

        pars_change_crop = {'oneOverFWhitening':
            {
                'central_clip': (64, 128),
                'n_jobs': -1,
            }
        }

        pars_change_crop_pure = {'oneOverFWhitening':
            {
                'central_clip': (64, 128),
                'no_filter': True,  # useful when only want to do central_clip,
                'n_jobs': -1,
            }
        }

        pipeline_default = pipeline.preprocessing_pipeline(steps_naive, pars_naive, order=steps_naive)[0]
        pipeline_change_f0 = pipeline.preprocessing_pipeline(steps_naive, pars_change_f0, order=steps_naive)[0]
        pipeline_cutoff = pipeline.preprocessing_pipeline(steps_naive, pars_no_cutoff, order=steps_naive)[0]
        pipeline_change_crop = pipeline.preprocessing_pipeline(steps_naive, pars_change_crop, order=steps_naive)[0]
        pipeline_change_crop_pure = pipeline.preprocessing_pipeline(steps_naive, pars_change_crop_pure,
                                                                    order=steps_naive)[0]

        def check_shape_and_close(X1, X2, no_filter=False):
            if not no_filter:
                # mean very close to zero
                self.assertTrue(abs(X2.mean()) < 1e-6)
            self.assertEqual(X1.shape, X2.shape)
            print(abs(X1 - X2).max())
            self.assertTrue(np.allclose(X1, X2))

        check_shape_and_close(new_images_default, pipeline_default.transform(original_images))
        check_shape_and_close(new_images_change_f0, pipeline_change_f0.transform(original_images))
        check_shape_and_close(new_images_no_cutoff, pipeline_cutoff.transform(original_images))
        check_shape_and_close(new_images_change_crop, pipeline_change_crop.transform(original_images))
        check_shape_and_close(new_images_change_crop_pure, pipeline_change_crop_pure.transform(original_images), True)


if __name__ == '__main__':
    unittest.main()
