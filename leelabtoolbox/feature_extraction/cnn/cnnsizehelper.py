"""size utility module for CNN. basically a rewrite of my old code to compute any size-related thing to CNN"""

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from collections import OrderedDict, defaultdict
from copy import deepcopy


def _check_exact_cover(output_size, stride, kernelsize, input_size, pad_this):
    output_size_np = np.asarray(output_size)
    input_size_np = np.asarray(input_size)
    assert np.all(
        (output_size_np - 1) * stride + kernelsize == input_size_np + 2 * pad_this), "input not exactly covered"


def _compute_minimum_coverage_helper(good_loc_min, good_loc_max):
    # it's either ordinary or too big.
    # first, check top_left_loc[0] and bottom_right_loc[0]
    if good_loc_min <= good_loc_max:
        # ordinary case
        res_loc_min = good_loc_min
        res_loc_max = good_loc_max + 1
    else:
        # the choice of neuron is ambiguous. simply pick the central one (well this central one is floored if
        # there's no exact central to be chosen).
        res_loc_min = (good_loc_min + good_loc_max) // 2
        res_loc_max = res_loc_min + 1

    return res_loc_min, res_loc_max


def _compute_minimum_coverage_helper_2(min_r_all, max_r_all, min_c_all, max_c_all, posr, posc):
    mask = np.logical_and(np.logical_and(min_r_all <= posr, posr < max_r_all),
                          np.logical_and(min_c_all <= posc, posc < max_c_all))
    return mask


def _check_layer_info_dict(layer_info_dict):
    assert isinstance(layer_info_dict, OrderedDict)
    for layer, value in layer_info_dict.items():
        assert {'pad', 'stride', 'kernelsize'} == set(value.keys())


class CNNSizeHelper(object):
    def __init__(self, layer_info_dict, input_size=None):
        """

        :param layer_info_dict:
        :param input_size:
        """
        _check_layer_info_dict(layer_info_dict)

        result_dict = OrderedDict()
        # input.
        last_stride_g = 1
        last_kernelsize_g = 1
        last_pad_g = 0  # cummulative padding.

        # this computes how many last layer's unit will units of shape k cover.
        # compositing functions for different layers will give you global kernelsize.

        kernel_func_dict = OrderedDict()
        last_kernel_func = lambda k: k

        def give_one_kernel_func(stride_local, kernelsize_local, last_func):
            def kernel_func_local(k):
                return last_func((k - 1) * stride_local + kernelsize_local)

            return kernel_func_local

        for layer, info_dict in layer_info_dict.items():
            # for each layer, I have to check that the units in the output blob
            # can perfectly cover the units in the current blob (no uncovered units, nor redundant receptive field).
            # this makes all computation much easier.

            # compute the stride, kernelsize, and pad for this layer, w.r.t. input layer.
            stride_this = info_dict['stride']
            kernelsize_this = info_dict['kernelsize']
            pad_this = info_dict['pad']

            pad_this_g = pad_this * last_stride_g + last_pad_g  # total padding in input layer
            stride_this_g = stride_this * last_stride_g  # total stride in input layer.
            kernelsize_this_g = (kernelsize_this - 1) * last_stride_g + last_kernelsize_g

            info_dict_out_this = deepcopy(info_dict)
            info_dict_out_this.update({
                'stride_g': stride_this_g,
                'kernelsize_g': kernelsize_this_g,
                'pad_g': pad_this_g
            })
            result_dict[layer] = info_dict_out_this

            # alternative to get kernel size
            # I didn't use lambda, due to scope issue
            # <http://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture-in-python>
            # this illustrates difference between passing argument and using closure. In passing argument, the object
            # that name points to is bound to a NEW variable; in closure, the object pointed to can change.
            this_kernel_func = give_one_kernel_func(stride_this, kernelsize_this, last_kernel_func)
            # verify kernelsize_g
            assert np.all(this_kernel_func(1) == kernelsize_this_g)
            # verify pad_g
            assert np.all(this_kernel_func(pad_this + 1) - this_kernel_func(1) == pad_this * last_stride_g)
            # still, I haven't verified the correctness of stride_g and pad_g very satisfactorily, but
            # I think it must be correct, given so many tests.
            kernel_func_dict[layer] = this_kernel_func

            last_kernel_func = this_kernel_func

            last_stride_g = stride_this_g
            last_kernelsize_g = kernelsize_this_g
            last_pad_g = pad_this_g

        self.layer_info_dict = result_dict
        self.kernel_func_dict = kernel_func_dict
        self.input_size = None
        if input_size is not None:
            self.compute_output_size(input_size)

    def compute_output_size(self, input_size):
        """give the blob size of each blob.

        Parameters
        ----------
        input_size:
            something of length 2, and all elements are integers.

        Returns
        -------

        """
        assert len(input_size) == 2, "you must specify both height and width"
        last_input_size = input_size
        for layer, info_dict_this in self.layer_info_dict.items():
            pad_this = info_dict_this['pad']
            stride_this = info_dict_this['stride']
            kernelsize_this = info_dict_this['kernelsize']

            pad_this_g = info_dict_this['pad_g']
            stride_this_g = info_dict_this['stride_g']
            kernelsize_this_g = info_dict_this['kernelsize_g']

            # compute the dimension of units for output.
            output_size = tuple((np.asarray(last_input_size) + 2 * pad_this - kernelsize_this) // stride_this + 1)

            # check that this output check can indeed exactly cover the input layer below,
            _check_exact_cover(output_size, stride_this, kernelsize_this, last_input_size, pad_this)
            # as well as input image (plus padding).
            _check_exact_cover(output_size, stride_this_g, kernelsize_this_g, input_size, pad_this_g)
            input_size_this = last_input_size
            input_size_this_g = tuple(np.asarray(input_size) + 2 * pad_this_g)

            info_dict_this.update({
                'output_size': output_size,
                'input_size': input_size_this,
                'input_size_g': input_size_this_g
            })

            last_input_size = output_size
        self.input_size = input_size

    def _compute_range_vector(self, layer_name, topvec, bottomvec, leftvec, rightvec):

        topvec = np.asarray(topvec)
        leftvec = np.asarray(leftvec)
        bottomvec = np.asarray(bottomvec)
        rightvec = np.asarray(rightvec)

        row_diff = bottomvec - topvec + 1
        col_diff = rightvec - leftvec + 1

        layer_info_dict_this = self.layer_info_dict[layer_name]
        kernelsize_g = layer_info_dict_this['kernelsize_g']
        stride_g = layer_info_dict_this['stride_g']
        pad_g = layer_info_dict_this['pad_g']
        assert np.all(topvec <= bottomvec) and np.all(leftvec <= rightvec) and np.all(topvec >= 0) and np.all(
            leftvec >= 0)

        min_r_all = stride_g * topvec - pad_g
        max_r_all = stride_g * bottomvec - pad_g + kernelsize_g
        min_c_all = stride_g * leftvec - pad_g
        max_c_all = stride_g * rightvec - pad_g + kernelsize_g

        ref_range_r = self.kernel_func_dict[layer_name](row_diff)
        ref_range_c = self.kernel_func_dict[layer_name](col_diff)

        # another way saying my kernel size computed is correct. plus that in my test script,
        # I tested that the position is correct. Thus everything is correct.
        assert np.array_equal(max_r_all - min_r_all, ref_range_r)
        assert np.array_equal(max_c_all - min_c_all, ref_range_c)

        return min_r_all, max_r_all, min_c_all, max_c_all

    def compute_range(self, layer_name, row_range, col_range):
        """computes the RF field coordinates in the original input space of a rectangle array of units
        may give negative values.
        everything here is left close right open, following Python slicing convention.
        """
        row_range = np.array(row_range, dtype=np.int64)
        col_range = np.array(col_range, dtype=np.int64)
        row_range[1] -= 1
        col_range[1] -= 1
        assert row_range.shape == col_range.shape == (2,)

        top_out, bottom_out, left_out, right_out = self._compute_range_vector(layer_name,
                                                                              *np.concatenate([row_range, col_range]))
        return (top_out, bottom_out), (left_out, right_out)

    def compute_inside_neuron(self, layer_name):
        """ compute the range of column and row indices that give a neuron whose RF is completely in the image.
        the ranges are given in Python convention, exclusive on the right part.

        """
        layer_info_dict_this = self.layer_info_dict[layer_name]

        field_size = layer_info_dict_this['kernelsize_g']
        stride = layer_info_dict_this['stride_g']
        pad = layer_info_dict_this['pad_g']

        # use np.ceil(pad / stride) to get index of leftmost/topmost unit that doesn't intersect with padding
        # use np.floor((input_rows - field_size + pad) / stride) to get last index of rightmost/bottommost unit
        # that has no intersection with padding.
        range_min = np.ceil(pad / stride).astype(np.int64)
        range_max = np.floor((np.asarray(self.input_size) - field_size + pad) / stride).astype(np.int64)
        range_min = np.broadcast_to(range_min, range_max.shape)
        if not np.all(range_min <= range_max):
            raise ValueError('No inside neuron!')
        return range_min[0], range_max[0] + 1, range_min[1], range_max[1] + 1

    def compute_minimum_coverage(self, layer_name, top_bottom, left_right):
        """ compute the miminum grid of neurons that can cover a rectangle with top left at top_left,
        and bottom right at bottom_right

        everything here is left close right open, following Python slicing convention.
        """
        # use int to avoid any potential problem. look that I floor the top left and ceil the bottom right to make sure
        # the actual image is covered, even with float input.
        top_pos, bottom_pos = top_bottom
        left_pos, right_pos = left_right
        top_pos = np.floor(top_pos).astype(np.int64)
        left_pos = np.floor(left_pos).astype(np.int64)
        bottom_pos = np.ceil(bottom_pos).astype(np.int64) - 1
        right_pos = np.ceil(right_pos).astype(np.int64) - 1

        assert 0 <= top_pos <= bottom_pos < self.input_size[0]
        assert 0 <= left_pos <= right_pos < self.input_size[1]

        # brutal force, find the four extreme neurons. It's fine, as we don't need performance here.
        output_size_this_layer = self.layer_info_dict[layer_name]['output_size']
        # get the coverage of all neurons.
        row_idx, col_idx = np.meshgrid(np.arange(output_size_this_layer[0]), np.arange(output_size_this_layer[1]),
                                       indexing='ij')

        # basically, I have output_size_this_layer.size cases, computing the coverage for each individual unit.

        (min_r_all, max_r_all,
         min_c_all, max_c_all) = self._compute_range_vector(layer_name, row_idx, row_idx, col_idx, col_idx)

        # top_left_loc is the bottom right unit that contains (top_pos, left_pos)
        mask_topleft = _compute_minimum_coverage_helper_2(min_r_all, max_r_all, min_c_all, max_c_all, top_pos, left_pos)
        top_left_loc = row_idx[mask_topleft].max(), col_idx[mask_topleft].max()
        # bottom_right_loc is the top left unit that contains (bottom_pos, right_pos)
        mask_bottomright = _compute_minimum_coverage_helper_2(min_r_all, max_r_all, min_c_all, max_c_all, bottom_pos,
                                                              right_pos)
        bottom_right_loc = row_idx[mask_bottomright].min(), col_idx[mask_bottomright].min()
        tl_loc_0, br_loc_0 = _compute_minimum_coverage_helper(top_left_loc[0], bottom_right_loc[0])
        tl_loc_1, br_loc_1 = _compute_minimum_coverage_helper(top_left_loc[1], bottom_right_loc[1])

        return (tl_loc_0, br_loc_0), (tl_loc_1, br_loc_1)


def create_size_helper(info_dict, input_size=None, last_layer=None):
    info_dict = deepcopy(info_dict)
    assert isinstance(info_dict, OrderedDict)
    if last_layer is not None:
        keys_all = list(info_dict.keys())
        first_key_to_remove_idx = keys_all.index(last_layer) + 1
        for layer_to_remove in keys_all[first_key_to_remove_idx:]:
            del info_dict[layer_to_remove]
    return CNNSizeHelper(info_dict, input_size)


def get_slice_dict(slice_dict, blobs_to_extract):
    # then, compute the slice dict
    if slice_dict is None:
        slice_dict = defaultdict(lambda: ((None, None), (None, None)))

    slice_dict_real = dict()

    for blob_name_to_read in blobs_to_extract:
        slice_exp_1, slice_exp_2 = slice_dict[blob_name_to_read]
        slice_r = slice(slice_exp_1[0], slice_exp_1[1])
        slice_c = slice(slice_exp_2[0], slice_exp_2[1])
        slice_dict_real[blob_name_to_read] = slice_r, slice_c
    return slice_dict_real
