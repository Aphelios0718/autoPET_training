import os
import json
from typing import Union
import numpy as np


def crop_ct(img):
    return img > -1024


def clip(ct_arr, lower: Union[int, float] = None, upper: Union[int, float] = None):
    lower = np.percentile(ct_arr, 0.5) if not lower else lower
    upper = np.percentile(ct_arr, 99.5) if not upper else upper
    return np.clip(ct_arr, lower, upper)


def semantic_seg_label(fpath):
    if "BENIGN" in fpath:
        return 1
    elif "LUNG_CANCER" in fpath:
        return 2
    elif "LYMPHOMA" in fpath:
        return 3


def read_splits_json(splits_json, data_root, dirname="npz_file", k=0, phase="train"):

    f = open(splits_json, "r")
    data_list = json.load(f)[k][phase]
    new_list = [f"{data_root}/{dirname}/{i}.npz" for i in data_list]
    assert os.path.exists(new_list[0])
    return new_list