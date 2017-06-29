from __future__ import absolute_import, division, print_function, unicode_literals

import os

# <http://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe>
os.environ['GLOG_minloglevel'] = '2'
from caffe.io import resize_image

import unittest
from itertools import product

import numpy as np

from leelabtoolbox.feature_extraction.cnn.caffe_network_definitions import (net_info_dict, proto_struct_dict,
                                                                            get_sub_prototxt_bytes, get_prototxt_bytes)
from leelabtoolbox.feature_extraction.cnn import caffe_preprocessing
from leelabtoolbox.feature_extraction.cnn import caffe_network_construction
from leelabtoolbox.feature_extraction.cnn import cnnsizehelper
from leelabtoolbox.feature_extraction.cnn import caffe_feature_extraction

rng_state = np.random.RandomState(seed=0)

# too big for travis
net_to_skip = {'vgg19'}

def _valid_input_selector(x, input_size):
    raw_r = np.arange(*x[0])
    raw_c = np.arange(*x[1])
    bool_r = np.logical_and(raw_r >= 0, raw_r < input_size[0])
    bool_c = np.logical_and(raw_c >= 0, raw_c < input_size[1])
    # 3 is for color image
    return np.ix_(np.arange(3), raw_r[bool_r], raw_c[bool_c])


def _fill_weight_random(net_this, rng_seed):
    rng_state = np.random.RandomState(seed=rng_seed)
    # let's fill the net with weights
    for layer_name, param in net_this.params.items():
        for idx in range(len(param)):
            param[idx].data[...] = rng_state.randn(*param[idx].data.shape) * 0.1


def _test_one_case_inside_neuron(test_instance, helper):
    counter = 0
    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0

    def _check_valid_range(range_min_max, input_size_one_side):
        range_min, range_max = range_min_max
        return 0 <= range_min < range_max <= input_size_one_side

    def compute_range_single(r, c):
        return helper.compute_range(layername, (r, r + 1), (c, c + 1))

    def check_ok(r_valid, c_valid, size_bound):
        test_instance.assertTrue(_check_valid_range(r_valid, size_bound[0]))
        test_instance.assertTrue(_check_valid_range(c_valid, size_bound[1]))

    def check_not_ok(r_valid, c_valid, size_bound):
        test_instance.assertFalse(
            _check_valid_range(r_valid, size_bound[0]) and _check_valid_range(c_valid, size_bound[1]))

    input_size = helper.input_size

    for layername in helper.layer_info_dict:
        try:
            row_lower, row_upper, col_lower, col_upper = helper.compute_inside_neuron(layername)
        except ValueError as e:
            if e.args[0] == 'No inside neuron!':
                print('this layer {} has no inside neuron'.format(layername))
                # raw_input('wait to continue')
                continue
            else:
                raise e
        counter += 1
        print(layername, row_lower, row_upper, col_lower, col_upper)
        # check that these bounds are consistent with the ones computed by compute_valid_area
        output_size_ref = helper.layer_info_dict[layername]['output_size']
        check_ok((row_lower, row_upper), (col_lower, col_upper), output_size_ref)
        # first check that neurons at (row_lower,col_lower) and (row_upper-1,col_upper-1) are with in RF.

        range_valid_r, range_valid_c = compute_range_single(row_lower, col_lower)
        check_ok(range_valid_r, range_valid_c, input_size)
        range_valid_r, range_valid_c = compute_range_single(row_upper - 1, col_upper - 1)
        check_ok(range_valid_r, range_valid_c, input_size)

        # then check that as long as we move out of the boundaries a little bit, we get invalid pixels.
        if row_lower > 0:
            counter1 += 1
            range_valid_r, range_valid_c = compute_range_single(row_lower - 1, col_lower)
            check_not_ok(range_valid_r, range_valid_c, input_size)

        if col_lower > 0:
            counter2 += 1
            range_valid_r, range_valid_c = compute_range_single(row_lower, col_lower - 1)
            check_not_ok(range_valid_r, range_valid_c, input_size)

        if row_upper < output_size_ref[0]:
            counter3 += 1
            range_valid_r, range_valid_c = compute_range_single(row_upper, col_upper - 1)
            check_not_ok(range_valid_r, range_valid_c, input_size)

        if col_upper < output_size_ref[1]:
            counter4 += 1
            range_valid_r, range_valid_c = compute_range_single(row_upper - 1, col_upper)
            check_not_ok(range_valid_r, range_valid_c, input_size)

    test_instance.assertTrue(counter > 0)
    test_instance.assertTrue(counter1 > 0)
    test_instance.assertTrue(counter2 > 0)
    test_instance.assertTrue(counter3 > 0)
    test_instance.assertTrue(counter4 > 0)


def test_size_util_minimum_coverage_helper(projection_this, layer, row_loc_old, col_loc_old, row_loc, col_loc):
    # I have to verify that this is correct.
    # first, check that the region covered from top_left_loc to bottom_right_loc contains my ROI.
    range_original_row, range_original_col = projection_this.compute_range(layer, row_loc, col_loc)
    # print('area1: {}, {}; area 2: {}, {}'.format(top_left, bottom_right, range_original_tl, range_original_br))
    result = check_one_area_inside_another(row_loc_old, col_loc_old, range_original_row, range_original_col)
    return result


def check_one_area_inside_another(row1, col1, row2, col2):
    """ check area 1 inside area 2.
    """
    row_flag = row2[0] <= row1[0] < row1[1] <= row2[1]
    col_flag = col2[0] <= col1[0] < col1[1] <= col2[1]

    return row_flag and col_flag


class MyTestCase(unittest.TestCase):
    def setUp(self):
        rng_state.seed(0)

    def test_network_definitions(self):
        # check that my sub prototxt works, at least for the trivial case.
        for net_to_test in net_info_dict:
            prototxt_bytes_ref = get_prototxt_bytes(net_info_dict[net_to_test]['prototxt_path'])
            prototxt_bytes_split_then_combined = get_sub_prototxt_bytes(proto_struct_dict[net_to_test])
            self.assertEqual(type(prototxt_bytes_ref), bytes)
            self.assertEqual(type(prototxt_bytes_split_then_combined), bytes)
            self.assertEqual(prototxt_bytes_ref, prototxt_bytes_split_then_combined)

    def test_caffe_preprocessing_transformer(self):
        input_name = 'test_blob'
        for _ in range(5):
            image_size_keep_list = [True, False]
            use_mu_list = [True, False]
            image_size = rng_state.randint(50, 100, size=(2,))
            num_image = rng_state.randint(1, 5 + 1)

            for image_size_keep, use_mu in product(image_size_keep_list, use_mu_list):
                print('keep size: {}, use mu: {}'.format(image_size_keep, use_mu))
                transformer_scale = rng_state.rand() + 0.1
                image_size_new = image_size.copy() if image_size_keep else (
                    image_size * (rng_state.rand(2) + 0.5)).astype(np.int64)
                print('image size raw {}, image size after {}'.format(image_size, image_size_new))

                if use_mu:
                    mu_actual = rng_state.randn(3).astype(np.float32)
                    mu_pass = mu_actual.copy()
                else:
                    mu_actual = caffe_preprocessing.imagenet_mu
                    mu_pass = None

                # let's process the images without transformer, step by step.
                raw_image_shape = num_image, image_size[0], image_size[1], 3
                blob_shape = num_image, 3, image_size_new[0], image_size_new[1]
                raw_images = rng_state.randn(*raw_image_shape).astype(np.float32)

                # first, resize the images if needed.
                if not image_size_keep:
                    blob_images_ref = np.asarray(
                        [resize_image(x, image_size_new) for x in raw_images]).astype(
                        np.float32)
                else:
                    blob_images_ref = raw_images.copy()

                # then transpose
                blob_images_ref = np.asarray([x.transpose((2, 0, 1)) for x in blob_images_ref])
                # then swap
                blob_images_ref = blob_images_ref[:, ::-1, :, :]
                # then scale
                blob_images_ref *= transformer_scale
                # then minus mean
                blob_images_ref -= mu_actual[np.newaxis, :, np.newaxis, np.newaxis]

                # ok. let's create the transformer
                transformer = caffe_preprocessing.create_transformer(input_name, blob_shape,
                                                                     scale=transformer_scale, mu=mu_pass)
                blob_images = caffe_preprocessing.transform_batch(transformer, raw_images, input_name)

                blob_var = blob_images_ref.var()
                self.assertTrue(np.all(np.isfinite(blob_images_ref)))
                self.assertTrue(np.all(np.isfinite(blob_var)))
                self.assertNotEqual(blob_var, 0)

                # if image_size_keep:
                self.assertTrue(np.array_equal(blob_images, blob_images_ref))

                # print('right!')
                # else:
                #     self.assertEqual(blob_images.shape, blob_images_ref.shape)
                #     self.assertEqual(blob_images.shape, blob_shape)
                #     print(abs(blob_images - blob_images_ref).max())
                #     print(pearsonr(blob_images.ravel(), blob_images_ref.ravel()))
                #     # this won't work due to caffe's way to deal with resizing. But in practice it shouldn't matter.
                #     self.assertTrue(np.allclose(blob_images, blob_images_ref, atol=1e-4))

    def test_sizehelper_blob_size_and_compute_range(self):
        # here, I will use a real CNN (alexnet) to verify that the range is computed corrrectly.
        # and verify that my blob sizes are computed correctly.
        for net_name in net_info_dict:
            net_info_this = net_info_dict[net_name]
            net_this = caffe_network_construction.create_predefined_net(net_name, load_weight=False)
            helper_this = cnnsizehelper.create_size_helper(net_info_this['conv_blob_info_dict'],
                                                           input_size=net_info_this['input_size'])
            input_blob = net_info_this['input_blob']
            shape_old = net_this.blobs[input_blob].data.shape
            shape_new = (1,) + shape_old[1:]
            net_this.blobs[input_blob].reshape(*shape_new)
            net_this.forward()
            last_input_size = net_info_this['input_size']
            for conv_blob_name, conv_blob_info in helper_this.layer_info_dict.items():
                print('{}: {}'.format(net_name, conv_blob_name))
                self.assertEqual((1,) + conv_blob_info['output_size'],
                                 net_this.blobs[conv_blob_name].data.shape[0:1] + net_this.blobs[
                                                                                      conv_blob_name].data.shape[2:])

                self.assertEqual(conv_blob_info['input_size'], last_input_size)
                last_input_size = conv_blob_info['output_size']

            # ok. now time to verify that the compute_range() is correct.
            # pick 4 corners and then the center one, each randomly from 1x1 to 3x3, which is always possible for the
            # example networks.

            # let's fill the net with weights
            for layer_name, param in net_this.params.items():
                for idx in range(len(param)):
                    param[idx].data[...] = rng_state.randn(*param[idx].data.shape) * 0.1

            # for each network, choose 5 blobs to test
            blobs_to_test = rng_state.choice(list(helper_this.layer_info_dict.keys()),
                                             size=5, replace=False)

            for conv_blob_name, conv_blob_info in helper_this.layer_info_dict.items():
                if conv_blob_name not in blobs_to_test:
                    continue
                print('test compute range, {}: {}'.format(net_name, conv_blob_name))
                h_o, w_o = conv_blob_info['output_size']
                # pick some areas to test.
                # I choose 5 areas, of same shape (size [1...h-1] x [1...w-1]),
                # at 4 corners, and center.
                area_height = rng_state.randint(1, h_o)
                area_width = rng_state.randint(1, w_o)
                area_1_location = (0, area_height), (0, area_width)
                area_2_location = (0, area_height), (w_o - area_width, w_o)
                area_3_location = (h_o - area_height, h_o), (0, area_width)
                area_4_location = (h_o - area_height, h_o), (w_o - area_width, w_o)
                area_5_location = ((h_o - area_height) // 2, (h_o + area_height) // 2), (
                    (w_o - area_width) // 2, (w_o + area_width) // 2)

                # over each area, I will first fill blob with random input,
                # then permute input apart from its input region, and then fetch result again.
                area_loc_this_layer_list = [area_1_location, area_2_location,
                                            area_3_location, area_4_location, area_5_location]
                area_loc_input_list = [helper_this.compute_range(conv_blob_name, *x) for x in area_loc_this_layer_list]
                area_loc_this_layer_selector_list = [
                    (slice(*x[0]), slice(*x[1])) for x in area_loc_this_layer_list
                    ]
                area_loc_input_selector_list = [
                    _valid_input_selector(x, net_info_this['input_size']) for x in area_loc_input_list
                    ]
                for area_sel, area_input_sel in zip(area_loc_this_layer_selector_list, area_loc_input_selector_list):
                    # first, get random data
                    # print(area_sel, area_input_sel)
                    net_this.blobs[input_blob].data[...] = rng_state.randn(*shape_new)
                    # then forward
                    net_this.forward()
                    # then fetch the data in input
                    data_keep_input = net_this.blobs[input_blob].data[0][area_input_sel].copy()
                    data_keep_output = net_this.blobs[conv_blob_name].data[:, :, area_sel[0], area_sel[1]].copy()
                    # now change data.
                    net_this.blobs[input_blob].data[...] = rng_state.randn(*shape_new)
                    net_this.forward()
                    data_keep_output_2_wrong = net_this.blobs[conv_blob_name].data[:, :, area_sel[0],
                                               area_sel[1]].copy()
                    # but put them back
                    # here I did it without using two index, avoiding possible assignment to temp array.
                    new_input = net_this.blobs[input_blob].data[0].copy()
                    new_input[area_input_sel] = data_keep_input
                    net_this.blobs[input_blob].data[0] = new_input
                    net_this.forward()
                    data_keep_output_2 = net_this.blobs[conv_blob_name].data[:, :, area_sel[0], area_sel[1]]

                    # TODO: for first layer, change pixels at 4 corners, and assert that the result is different.
                    # it's difficult to do for higher layers, since there are stuffs like pooling, etc.
                    # well, let's not do it, but try another analytical approach.

                    self.assertTrue(np.array_equal(data_keep_output, data_keep_output_2))
                    self.assertFalse(np.array_equal(data_keep_output, data_keep_output_2_wrong))

                    for bb in [data_keep_output, data_keep_output_2, data_keep_output_2_wrong]:
                        bb_var = bb.var()
                        self.assertTrue(np.all(np.isfinite(bb)))
                        self.assertTrue(np.all(np.isfinite(bb_var)))
                        self.assertNotEqual(bb_var, 0)

    def test_sizehelper_inside(self):
        for net_name in net_info_dict:
            print(net_name)
            net_info_this = net_info_dict[net_name]
            helper_this = cnnsizehelper.create_size_helper(net_info_this['conv_blob_info_dict'],
                                                           input_size=net_info_this['input_size'])
            _test_one_case_inside_neuron(self, helper_this)

    def test_sizehelper_minimum(self):
        for model in net_info_dict:
            net_info_this = net_info_dict[model]
            input_size = net_info_dict[model]['input_size']
            projection_this = cnnsizehelper.create_size_helper(net_info_this['conv_blob_info_dict'],
                                                               input_size=input_size)
            print('testing model {}'.format(model))
            counter1 = 0
            counter2 = 0
            for layer in projection_this.layer_info_dict:
                print('testing model {}, layer {}'.format(model, layer))
                for idx in range(10):
                    print('iter {}'.format(idx + 1))
                    center = np.random.randint(100, input_size[0] - 100, size=(2,))
                    image_size = np.random.randint(50, 100)
                    # only square stuff tested yet.
                    # I deliberately use /, not //
                    top_left = center - image_size / 2
                    bottom_right = center + image_size / 2
                    row_smaller = (top_left[0], bottom_right[0])
                    col_smaller = (top_left[1], bottom_right[1])

                    row_loc, col_loc = projection_this.compute_minimum_coverage(layer, (top_left[0], bottom_right[0]),
                                                                                (top_left[1], bottom_right[1]))

                    row_loc = np.asarray(row_loc, dtype=int)
                    col_loc = np.asarray(col_loc, dtype=int)

                    self.assertTrue(
                        test_size_util_minimum_coverage_helper(projection_this, layer, row_smaller, col_smaller,
                                                               row_loc, col_loc))
                    # try removing one row
                    if row_loc[1] - row_loc[0] > 1:
                        self.assertFalse(
                            test_size_util_minimum_coverage_helper(projection_this, layer, row_smaller, col_smaller,
                                                                   row_loc + np.array([1, 0]), col_loc))
                        self.assertFalse(
                            test_size_util_minimum_coverage_helper(projection_this, layer, row_smaller, col_smaller,
                                                                   row_loc - np.array([0, 1]), col_loc))
                        counter1 += 1
                    if col_loc[1] - col_loc[0] > 1:
                        self.assertFalse(
                            test_size_util_minimum_coverage_helper(projection_this, layer, row_smaller, col_smaller,
                                                                   row_loc, col_loc + np.array([1, 0])))
                        self.assertFalse(
                            test_size_util_minimum_coverage_helper(projection_this, layer, row_smaller, col_smaller,
                                                                   row_loc, col_loc - np.array([0, 1])))
                        counter2 += 1
            self.assertTrue(counter1 > 0)
            self.assertTrue(counter2 > 0)

    def test_sizehelper_with_last_layer(self):
        # here, I will use a real CNN (alexnet) to verify that the range is computed corrrectly.
        # and verify that my blob sizes are computed correctly.
        for net_name in net_info_dict:
            net_info_this = net_info_dict[net_name]
            helper_this = cnnsizehelper.create_size_helper(net_info_this['conv_blob_info_dict'],
                                                           input_size=net_info_this['input_size'],
                                                           last_layer='pool5')
            helper_this_2 = cnnsizehelper.create_size_helper(net_info_this['conv_blob_info_dict'],
                                                             input_size=net_info_this['input_size'])
            self.assertEqual(helper_this.input_size, helper_this_2.input_size)
            self.assertEqual(helper_this.layer_info_dict, helper_this_2.layer_info_dict)

            # another is random one. dropping some layer
            for _ in range(10):
                layer_to_keep = rng_state.choice(list(net_info_this['conv_blob_info_dict'].keys()))
                print('last layer {}/{}'.format(net_name, layer_to_keep))
                helper_this = cnnsizehelper.create_size_helper(net_info_this['conv_blob_info_dict'],
                                                               input_size=net_info_this['input_size'],
                                                               last_layer=layer_to_keep)
                key_index = list(net_info_this['conv_blob_info_dict'].keys()).index(layer_to_keep)
                key_ref = list(net_info_this['conv_blob_info_dict'].keys())[:key_index + 1]
                self.assertEqual(helper_this.input_size, helper_this_2.input_size)
                self.assertLessEqual(set(helper_this.layer_info_dict.keys()), set(key_ref))
                for x in helper_this.layer_info_dict:
                    self.assertEqual(helper_this.layer_info_dict[x],
                                     helper_this_2.layer_info_dict[x])

    def test_fill_cnn(self):
        # here, I will use a real CNN (alexnet) to verify that the range is computed corrrectly.
        # and verify that my blob sizes are computed correctly.
        for net_name in net_info_dict:
            print(net_name)
            net_info_this = net_info_dict[net_name]
            net_this = caffe_network_construction.create_predefined_net(net_name, load_weight=False)
            net_this_2 = caffe_network_construction.create_predefined_net(net_name, load_weight=False)
            _fill_weight_random(net_this, rng_seed=0)
            caffe_network_construction.fill_weights(net_this, net_this_2)
            # fill net_this again
            _fill_weight_random(net_this, rng_seed=1)
            # check they are all different
            for layer_name, param in net_this_2.params.items():
                for idx in range(len(param)):
                    self.assertFalse(np.array_equal(param[idx].data, net_this.params[layer_name][idx].data))
            # fill again
            _fill_weight_random(net_this, rng_seed=0)
            for layer_name, param in net_this_2.params.items():
                for idx in range(len(param)):
                    self.assertFalse(np.may_share_memory(param[idx].data, net_this.params[layer_name][idx].data))
                    self.assertTrue(np.array_equal(param[idx].data, net_this.params[layer_name][idx].data))

    def test_convert_slice_dict(self):

        for _ in range(100):
            # first, randomly generate some blob names
            # as well as their associate slice
            # for convenience, I will only test slice of form slice(x,y) where x and y are both ingeters.
            # and x < y.
            blobnames = [x for x in 'abcdefghijklmnopqrstuvwxyz']
            assert len(blobnames) == len(set(blobnames))  # unique
            blobnames_to_extract = rng_state.choice(blobnames, size=5, replace=False)
            # ok, create pairing slices.
            row_start_all = rng_state.randint(0, 20, len(blobnames))
            col_start_all = rng_state.randint(0, 20, len(blobnames))
            row_end_all = row_start_all + rng_state.randint(0, 20, len(blobnames))
            col_end_all = col_start_all + rng_state.randint(0, 20, len(blobnames))
            slice_dict = dict()
            slice_dict_correct = dict()
            slice_dict_correct_none = dict()
            assert len(row_start_all) == len(row_end_all) == len(col_start_all) == len(
                col_end_all) == len(blobnames)
            for (blob, row_start, row_end, col_start, col_end) in zip(blobnames, row_start_all, row_end_all,
                                                                      col_start_all, col_end_all):
                slice_dict[blob] = (row_start, row_end), (col_start, col_end)
                if blob in blobnames_to_extract:
                    slice_dict_correct[blob] = slice(row_start, row_end), slice(col_start, col_end)
                    slice_dict_correct_none[blob] = slice(None), slice(None)

            slice_dict_computed = cnnsizehelper.get_slice_dict(slice_dict, blobnames_to_extract)
            self.assertEqual(slice_dict_correct, slice_dict_computed)
            slice_dict_computed_none = cnnsizehelper.get_slice_dict(None, blobnames_to_extract)
            self.assertEqual(slice_dict_correct_none, slice_dict_computed_none)

    def test_caffe_feature_extraction(self):
        for net_name in net_info_dict:
            if net_name in net_to_skip:
                continue
            print(net_name)
            net_info_this = net_info_dict[net_name]
            net_this = caffe_network_construction.create_predefined_net(net_name, load_weight=False)
            net_this_2 = caffe_network_construction.create_predefined_net(net_name, load_weight=False)
            _fill_weight_random(net_this, rng_seed=0)
            _fill_weight_random(net_this_2, rng_seed=0)

            for layer_name, param in net_this_2.params.items():
                for idx in range(len(param)):
                    self.assertFalse(np.may_share_memory(param[idx].data, net_this.params[layer_name][idx].data))
                    self.assertTrue(np.array_equal(param[idx].data, net_this.params[layer_name][idx].data))

            # ok. let's reshape them.
            num_image = 6  # this is fine
            batch_size_list = [2, 3, 4]

            input_blob_shape = (num_image, 3) + net_info_this['input_size']
            raw_input_blob = (rng_state.randn(*input_blob_shape)).astype(np.float32)
            input_blob = net_info_this['input_blob']
            shape_old = net_this.blobs[input_blob].data.shape
            shape_new = (1,) + shape_old[1:]
            assert shape_old[1:] == input_blob_shape[1:]
            net_this.blobs[input_blob].reshape(*shape_new)

            use_none_slice_flag = False

            net_this.forward()
            # generate random slice shapes for each layer.
            blobs_to_extract = list(set(net_this.blobs.keys()) - {input_blob})
            slice_dict = dict()
            for blob in blobs_to_extract:
                original_blob_shape = net_this.blobs[blob].data.shape
                if len(original_blob_shape) == 2:
                    slice_this = slice(None), slice(None)
                else:
                    assert len(original_blob_shape) == 4
                    h, w = original_blob_shape[2:]
                    slice_row = slice(*np.sort(rng_state.choice(np.arange(0, h + 1), size=2, replace=False)))
                    slice_col = slice(*np.sort(rng_state.choice(np.arange(0, w + 1), size=2, replace=False)))
                    if rng_state.rand() > 0.5:
                        use_none_slice_flag = True
                        slice_this = slice(None), slice(None)
                    else:
                        slice_this = slice_row, slice_col
                slice_dict[blob] = slice_this
            print(slice_dict)

            assert use_none_slice_flag

            # first, extract all the ones
            net_this.blobs[input_blob].reshape(*input_blob_shape)
            net_this.blobs[input_blob].data[...] = raw_input_blob
            self.assertFalse(np.may_share_memory(net_this.blobs[input_blob].data,
                                                 raw_input_blob))
            net_this.forward()
            ref_extracted_features = dict()
            for blob in blobs_to_extract:
                blob_whole = net_this.blobs[blob].data
                if blob_whole.ndim == 2:
                    blob_whole = blob_whole[:, :, np.newaxis, np.newaxis]
                assert blob_whole.ndim == 4
                ref_extracted_features[blob] = blob_whole[:, :, slice_dict[blob][0],
                                               slice_dict[blob][1]]
                if slice_dict[blob][0].start is not None:
                    assert ref_extracted_features[blob].shape[2:] == (
                        slice_dict[blob][0].stop - slice_dict[blob][0].start,
                        slice_dict[blob][1].stop - slice_dict[blob][
                            1].start)
                # can't be constant.
                blob_var = blob_whole.var()
                print(blob, 'var', blob_var)
                self.assertTrue(np.all(np.isfinite(blob_whole)))
                self.assertTrue(np.all(np.isfinite(blob_var)))
                self.assertNotEqual(blob_var, 0)

            for batch_size in batch_size_list:
                print('batch size {}'.format(batch_size))
                computed_extracted_features = caffe_feature_extraction.extract_features(net_this_2,
                                                                                        [raw_input_blob],
                                                                                        input_blobs=[input_blob],
                                                                                        blobs_to_extract=blobs_to_extract,
                                                                                        batch_size=batch_size,
                                                                                        slice_dict=slice_dict)
                self.assertEqual(set(computed_extracted_features.keys()), set(ref_extracted_features.keys()))
                self.assertEqual(set(computed_extracted_features.keys()), set(blobs_to_extract))

                for blob in computed_extracted_features:
                    blob1 = computed_extracted_features[blob]
                    blob2 = ref_extracted_features[blob]
                    self.assertTrue(np.array_equal(blob1, blob2))
                    self.assertFalse(np.may_share_memory(blob1, blob2))

                # another one is full one.
                del computed_extracted_features
                computed_extracted_features_none = caffe_feature_extraction.extract_features(net_this_2,
                                                                                             [raw_input_blob],
                                                                                             input_blobs=None,
                                                                                             blobs_to_extract=None,
                                                                                             batch_size=batch_size,
                                                                                             slice_dict=None)
                self.assertEqual(set(computed_extracted_features_none.keys()), set(blobs_to_extract))

                for blob in computed_extracted_features_none:
                    blob1 = computed_extracted_features_none[blob]
                    blob2 = net_this.blobs[blob].data
                    if blob2.ndim == 2:
                        blob2 = blob2[:, :, np.newaxis, np.newaxis]
                    self.assertTrue(np.array_equal(blob1, blob2))
                    self.assertFalse(np.may_share_memory(blob1, blob2))
                del computed_extracted_features_none

    def test_caffe_feature_extraction_from_smaller_ones(self):
        for net_name in net_info_dict:
            if net_name in net_to_skip:
                continue
            print(net_name)
            net_info_this = net_info_dict[net_name]
            net_this = caffe_network_construction.create_predefined_net(net_name, load_weight=False)
            # choose a top layer.
            top_layer = rng_state.choice(list(net_info_this['num_layer_by_blob_dict'].keys()))
            print('up to {}'.format(top_layer))
            net_this_smaller_prototxt = get_sub_prototxt_bytes(proto_struct_dict[net_name], last_blob=top_layer)
            net_this_smaller = caffe_network_construction.create_empty_net(net_this_smaller_prototxt)

            _fill_weight_random(net_this, rng_seed=0)
            caffe_network_construction.fill_weights(net_this, net_this_smaller)

            for layer_name, param in net_this_smaller.params.items():
                for idx in range(len(param)):
                    self.assertFalse(np.may_share_memory(param[idx].data, net_this.params[layer_name][idx].data))
                    self.assertTrue(np.array_equal(param[idx].data, net_this.params[layer_name][idx].data))

            # ok. let's reshape them.
            num_image = 2  # this is fine

            input_blob_shape = (num_image, 3) + net_info_this['input_size']
            raw_input_blob = (rng_state.randn(*input_blob_shape)).astype(np.float32)
            input_blob = net_info_this['input_blob']
            shape_old = net_this.blobs[input_blob].data.shape
            assert shape_old[1:] == input_blob_shape[1:]
            net_this.blobs[input_blob].reshape(*input_blob_shape)
            net_this_smaller.blobs[input_blob].reshape(*input_blob_shape)
            net_this.blobs[input_blob].data[...] = raw_input_blob
            net_this_smaller.blobs[input_blob].data[...] = raw_input_blob
            self.assertFalse(
                np.may_share_memory(net_this_smaller.blobs[input_blob].data, net_this.blobs[input_blob].data))
            self.assertTrue(np.array_equal(net_this_smaller.blobs[input_blob].data, net_this.blobs[input_blob].data))
            # ok, forward both
            net_this_smaller.forward()
            net_this.forward()

            # ok. let's compare their blobs
            for blob in net_this_smaller.blobs:
                print('check {}'.format(blob))
                blob1 = net_this_smaller.blobs[blob].data
                blob2 = net_this.blobs[blob].data
                blob_var = blob1.var()
                print(blob1.shape, blob_var)
                self.assertTrue(np.all(np.isfinite(blob1)))
                self.assertTrue(np.all(np.isfinite(blob_var)))
                self.assertNotEqual(blob_var, 0)
                self.assertFalse(np.may_share_memory(blob1, blob2))
                self.assertTrue(np.array_equal(blob1, blob2))


if __name__ == '__main__':
    unittest.main(failfast=True)
