from __future__ import absolute_import, division, print_function, unicode_literals

from copy import deepcopy

from sklearn.pipeline import Pipeline

from .transformers import transformer_dict, default_pars_dict

_preprocessing_pipeline_all_steps = default_pars_dict.keys()


def preprocessing_pipeline(steps=None, pars=None, order=None):
    # process steps
    if steps is None:
        raise NotImplementedError('no default steps implemented yet!')

    steps = frozenset(steps)
    assert steps <= _preprocessing_pipeline_all_steps, "there are undefined operations!"

    # process pars
    default_pars = deepcopy(default_pars_dict)
    if pars is None:
        pars = default_pars
    else:
        # since you specify keys yourself, I assume that you are only modifying relevant pars.
        assert frozenset(pars.keys()) <= steps, "you can't define pars for steps not in the pipeline!"
    # construct a pars with only relevant steps.
    real_pars = {key: default_pars[key] for key in steps}
    for key in pars:
        real_pars[key].update(pars[key])

    # process order
    if order is None:
        raise NotImplementedError('no default order implemented yet!')

    pipeline_step_list = []

    for candidate_step in order:
        if candidate_step in steps:
            pipeline_step_list.append((candidate_step,
                                       transformer_dict[candidate_step](real_pars[candidate_step])))

    return Pipeline(pipeline_step_list), deepcopy(real_pars)
