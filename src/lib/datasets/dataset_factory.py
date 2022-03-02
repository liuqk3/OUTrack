from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde_json import JointDataset as JointDatasetJson


def get_dataset(dataset, task):
    assert task == 'mot'
    if dataset == 'jde':
        raise RuntimeError("This is the original FairMOT code, and is not needed")
    elif dataset == 'jde_json':
        return JointDatasetJson
    else:
        raise ValueError(dataset)