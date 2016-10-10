from __future__ import absolute_import, division, print_function, unicode_literals

from functools import partial

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from ..util import normalize_random_state, make_2d_array

# turn off all validation.
FunctionTransformer = partial(FunctionTransformer, validate=False)

# TODO do parameter check in the beginning when transoformer is generated, not when using applying transformer.

# these two saves the default pars and how to generate transformers.
transformer_dict = {}
default_pars_dict = {}


def register_transformer(name, func, default_pars=None, pars_validator=None):
    # TODO: people should register a param checker as well. default checker should only accept `{}`.
    if default_pars is None:
        default_pars = {}
    if pars_validator is None:
        # by default, only check that keys in passed pars match keys in default_pars.
        pars_validator = lambda pars: set(pars.keys()) == set(default_pars.keys())

    def func_for_this_transform(pars):
        assert pars_validator(pars)
        return func(pars)

    transformer_dict[name] = func_for_this_transform
    default_pars_dict[name] = default_pars


def log_transform(images, bias=1.0, scale_factor=1.0, verbose=False):
    """do log transformation on images, pixel wise

    Log transformation will compress the high values, expand the low values, which is usually done when encoding
    luminance to 8-bit grayscale. This is useful when processing some images in RAW format.

    Parameters
    ----------
    images: np.ndarray
        images to be converted. images + bias should be > 0 (not checked)
    bias: float
        bias added to images. by default 1.0, which is typical for log transform, as in
        http://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm or
        https://www.tutorialspoint.com/dip/gray_level_transformations.htm
    scale_factor: float
        scale factor after taking log. usually not very useful in practice, as a constant scaling doesn't matter.
    verbose: bool
        whether print more info.

    Returns
    -------

    """
    if verbose:
        print("doing log transform with bias {}, scale_factor {}...".format(bias, scale_factor))
    # use log1p to be in theory more accurate.
    return scale_factor * np.log1p(np.asarray(images) + bias - 1)


def gamma_transform(images, gamma=0.5, scale_factor=1.0, verbose=False):
    """do gamma transformation on images.

    Parameters
    ----------
    images: np.ndarray
        images to be converted. should be >= 0 (not checked)
    gamma: float
        exponent for transformation. should be > 0. (not checked)
        a factor < 1 will compress the high values, expand the low values, which is usually done when encoding
        luminance to 8-bit grayscale. This is useful when processing some images in RAW format.
        TODO add more doc on when a > 1 gamma will be useful.
    scale_factor: float
        scale factor after taking gamma. usually not very useful in practice, as a constant scaling doesn't matter.
    verbose: bool
        whether print more info.

    Returns
    -------

    """
    if verbose:
        print("doing gamma transform with gamma {}, scale_factor {}".format(gamma, scale_factor))
    return scale_factor * np.power(np.asarray(images), gamma)


def _sampling_random_handle_pars(n_img, patchsize, numpatches, buff, seed, fixed_locations):
    rng_state = normalize_random_state(seed)
    if fixed_locations is not None:
        fixed_locations_flag = True
        if len(fixed_locations) == 1:
            fixed_locations_single = True
        elif len(fixed_locations) == n_img:
            fixed_locations_single = False
        else:
            raise ValueError('either one fixed_location, or `n_img` of them')
    else:
        fixed_locations_flag = False
        fixed_locations_single = True  # just convention.

    if not fixed_locations_flag:
        sample_per_image = int(np.floor(numpatches / n_img))
    else:
        sample_per_image = None

    patchsize_h, patchsize_w = np.broadcast_to(patchsize, (2,))
    buff_h, buff_w = np.broadcast_to(buff, (2,))

    return (rng_state, sample_per_image,
            patchsize_h, patchsize_w, buff_h, buff_w,
            fixed_locations_flag, fixed_locations_single)


def _sampling_random_get_locations(idx, image, n_img, sample_per_image, numpatches, rng_state,
                                   patchsize_h, patchsize_w, buff_h, buff_w):
    height, width = image.shape[:2]
    # determine how many points to sample.
    if idx + 1 < n_img:
        sample_this_image = sample_per_image
    else:
        assert idx + 1 == n_img
        sample_this_image = numpatches - idx * sample_per_image
        assert sample_this_image >= sample_per_image
    locations_this = np.zeros((sample_this_image, 2), dtype=np.uint16)
    h_max = height - 2 * buff_h - patchsize_h
    w_max = width - 2 * buff_w - patchsize_w
    for idx, (buff_this, max_this) in enumerate(zip((buff_h, buff_w), (h_max, w_max))):
        locations_this[:, idx] = buff_this + rng_state.randint(low=0, high=max_this + 1, size=(sample_this_image,))
    return locations_this


def sampling_random(images, patchsize, numpatches, buff=0, seed=None,
                    fixed_locations=None, return_locations=False, verbose=False):
    """

    Parameters
    ----------
    images: list of np.ndarray of at least 2d
        shapes for elements in images must be of same ndim, but can have different shape in the first 2 dimensions.
    patchsize: int
        how big the patch should be. can be a int or a tuple of ints, specifying (height, width) of patch.
    numpatches: int
        how many patches you want to extract in total.
    buff: int
        a padding factor around image to avoid some artifact or border of images.
    fixed_locations: list of np.ndarray
        a list of locations of top left corners for images of each patch. if this is specified, then
        no sampling will happen, as `fixed_locations` have already provided sampling.
    return_locations: bool
        whether return a the sampled locations in the same format of `fixed_locations`.
    verbose:
        whether print more info.

    Returns
    -------

    """
    new_image_list = []
    location_list = []
    n_img = len(images)

    (rng_state, sample_per_image,
     patchsize_h, patchsize_w, buff_h, buff_w,
     fixed_locations_flag, fixed_locations_single) = _sampling_random_handle_pars(n_img, patchsize, numpatches, buff,
                                                                                  seed, fixed_locations)

    for idx, image in enumerate(images):
        if verbose:
            print("[{}/{}]".format(idx + 1, n_img))
        if fixed_locations_flag:
            locations_this = fixed_locations[0] if fixed_locations_single else fixed_locations[idx]
        else:
            locations_this = _sampling_random_get_locations(idx, image, n_img, sample_per_image, numpatches, rng_state,
                                                            patchsize_h, patchsize_w, buff_h, buff_w)

        # do patch extraction
        assert locations_this.ndim == 2 and locations_this.shape[1] == 2
        if return_locations:
            location_list.append(np.array(locations_this))  # copy it, and then make sure it's base array

        for loc in locations_this:
            patch_this = image[loc[0]:loc[0] + patchsize_h, loc[1]:loc[1] + patchsize_w]
            new_image_list.append(patch_this)
    result = np.array(new_image_list)  # return as a 3d array.
    if verbose:
        print("sampled shape: ", result.shape)
    if not return_locations:
        return result
    else:
        assert len(location_list) == len(images)
        return result, location_list  # second argument being location list of list


def sampling_transformer(step_pars):
    sampling_type = step_pars['type']
    patchsize = step_pars['patchsize']
    if sampling_type == 'random' or sampling_type == 'fixed':
        # return_locations is not used in this transformer API.
        return FunctionTransformer(partial(sampling_random, patchsize=patchsize,
                                           numpatches=step_pars['random_numpatch'],
                                           seed=step_pars['random_seed'],
                                           fixed_locations=step_pars['fixed_locations'],
                                           verbose=step_pars['verbose'],
                                           buff=step_pars['random_buff']))
    else:
        raise NotImplementedError("type {} not supported!".format(sampling_type))


def _get_simple_transformer(func):
    return (lambda pars: FunctionTransformer(partial(func, **pars)))


register_transformer('gammaTransform', _get_simple_transformer(gamma_transform),
                     {'gamma': 0.5, 'scale_factor': 1.0, 'verbose': False})
register_transformer('logTransform', _get_simple_transformer(log_transform),
                     {'bias': 1, 'scale_factor': 1.0, 'verbose': False})
register_transformer('sampling', sampling_transformer,
                     {
                         'type': 'random',
                         'patchsize': None,
                         'random_numpatch': None,  # only for random
                         'random_buff': 0,  # only for random
                         'random_seed': None,
                         'fixed_locations': None,  # should be an iterable of len 1 or len of images, each
                         # being a n_patch x 2 array telling the row and column of top left corner.
                         'verbose': True
                     })
register_transformer('removeDC', lambda _: FunctionTransformer(lambda x: x - np.mean(x, axis=1, keepdims=True)))
register_transformer('flattening', lambda _: FunctionTransformer(make_2d_array))
