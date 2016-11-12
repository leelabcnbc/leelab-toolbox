"""basically a Python implementation of https://github.com/rsagroup/rsatoolbox
I implemented it according to <http://www.mrc-cbu.cam.ac.uk/methods-and-resources/toolboxes/license/>,
and not sure if it changed or not, and whether it's different from the one on GitHub.
the one `rsatoolbox.zip` I tested against has SHA1 00b3b3b2c11f03895c56099ebd53a5e568f2ef3a
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import pairwise_distances
from joblib import Parallel, delayed

from .util import normalize_random_state


def compute_rdm(X, noise_level=0.0, rng_state=None, method='scipy'):
    """
    Parameters
    ----------
    X: ndarray
    noise_level
    rng_state
    method
    Returns
    -------
    """
    assert X.ndim == 2
    rng_state = normalize_random_state(rng_state)
    _X = (X + noise_level * rng_state.randn(*X.shape)) if (noise_level != 0.0) else X
    if method == 'scipy':
        result = pdist(_X, metric='correlation')
    elif method == 'sklearn':
        # parallel.
        result = pairwise_distances(_X, metric='correlation', n_jobs=-1)
        if not np.allclose(result, result.T):
            print(abs(result - result.T).max())
            raise RuntimeError('too big error!')
        result = squareform(result, checks=False)
    else:
        raise RuntimeError('no such method!')
    assert np.all(np.isfinite(result)), "I can't allow invalid values in RDM"
    return result


def _rdm_similarity_check_type(ref_rdms, model_rdms, similarity_type):
    assert similarity_type == 'spearman', 'no other similarity type than spearman supported!'
    ref_rdms = np.asarray(ref_rdms)
    assert ref_rdms.ndim == 2
    return ref_rdms, model_rdms
    # I don't touch model_rdms, which can be things like HDF5 iterator.


def _rdm_similarity_one_case(ref_rdms_ranked, rdm):
    rdm = np.asarray(rdm)
    assert rdm.shape == (ref_rdms_ranked.shape[1],)
    # then perform cdist
    rdm_ranked = rankdata(rdm)[np.newaxis]
    assert rdm_ranked.shape == (1, ref_rdms_ranked.shape[1])
    return 1 - cdist(rdm_ranked, ref_rdms_ranked, metric='correlation')


def rdm_similarity(ref_rdms, model_rdms, similarity_type='spearman', n_jobs=-1, verbose=0):
    """
    Parameters
    ----------
    similarity_type : str or unicode
        can only be 'spearman' now.
    ref_rdms: ndarray. must be 2d
    model_rdms: ndarray. must be 2d
    Returns
    -------
    """
    ref_rdms, model_rdms = _rdm_similarity_check_type(ref_rdms, model_rdms, similarity_type)
    # do rank transform first, and then compute pearson.

    ref_rdms_ranked = np.asarray([rankdata(ref_rdm_this) for ref_rdm_this in ref_rdms])
    rdm_similarities = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_rdm_similarity_one_case)(ref_rdms_ranked, rdm) for rdm in model_rdms)
    assert rdm_similarities  # can't be empty
    rdm_similarities = np.concatenate(rdm_similarities, axis=0)
    assert rdm_similarities.ndim == 2

    return rdm_similarities


def _compute_rdm_bounds_upper(rdm_list, similarity_type, legacy, n_jobs):
    if similarity_type == 'spearman':
        # transform everything to rank
        rdm_list_rank = np.empty_like(rdm_list)
        for idx, rdm in enumerate(rdm_list):
            rdm_list_rank[idx] = rankdata(rdm)
        if legacy:  # use rank transformed data, even for lower bound.
            rdm_list = rdm_list_rank
        best_rdm = rdm_list_rank.mean(axis=0)
        # I can also do
        upper_bound = rdm_similarity(best_rdm[np.newaxis], rdm_list, similarity_type='spearman',
                                     n_jobs=n_jobs).mean()
    else:
        raise ValueError('supported similarity type!')

    return upper_bound, rdm_list


def _compute_rdm_bounds_lower(rdm_list, similarity_type):
    assert similarity_type == 'spearman'
    # compute lower bound. cross validation.
    # maybe it's good to use np.bool_ rather than np.bool
    # check <https://github.com/numba/numba/issues/1311>
    n_rdm = len(rdm_list)
    all_true_vector = np.ones((n_rdm,), dtype=np.bool_)
    similarity_list_for_lowerbound = []
    for i_rdm in range(n_rdm):
        selection_vector_this = all_true_vector.copy()
        selection_vector_this[i_rdm] = False
        similarity_list_for_lowerbound.append(spearmanr(rdm_list[selection_vector_this].mean(axis=0),
                                                        rdm_list[i_rdm]).correlation)
    lower_bound = np.array(similarity_list_for_lowerbound).mean()
    return lower_bound


def compute_rdm_bounds_check_input(rdm_list):
    rdm_list = np.asarray(rdm_list)
    assert rdm_list.ndim == 2
    n_rdm = rdm_list.shape[0]
    assert n_rdm >= 3, 'at least 3 RDMs to compute bounds (with 2, cross validation cannot be done)'

    return rdm_list


def compute_rdm_bounds(rdm_list, similarity_type='spearman', legacy=True, n_jobs=1):
    """computes the estimated lower and upper bounds of the similarity between the ideal model and the given data..
    this is a remake of ``ceilingAvgRDMcorr`` in rsatoolbox in MATLAB.
    Parameters
    ----------
    legacy : bool
        whether behave exactly the same as in rsatoolbox. maybe this is the correct behavior.
        so basically, in lower bound computation, at each time, we compute the best rdm for all but one RDMs,
        and we hope this RDM to under fit. (I don't know whether one rdm's difference will turn overfit to underfit).
        In practice, this seems to make little difference.

        this legacy behavior, while it's computing the optimal rdm for all but 1, it's not the case for other types,
        such as Kendall, whose upper bound is computed using an iterative approach, yet for lower bound, it's now.
    rdm_list
    similarity_type : str or unicode
        type of similarity. only spearman is supported currently.
    Returns
    -------
    """
    rdm_list = compute_rdm_bounds_check_input(rdm_list)
    upper_bound, rdm_list = _compute_rdm_bounds_upper(rdm_list, similarity_type, legacy, n_jobs)
    lower_bound = _compute_rdm_bounds_lower(rdm_list, similarity_type)

    return lower_bound, upper_bound
