import os
import random
import numpy as np
import pandas as pd
from typing import Union, Sequence


def get_relative_path(cur_dir, base_dir):
    relative_path = os.path.relpath(cur_dir, base_dir)
    assert os.path.isfile(relative_path)
    return relative_path
     
    
def split_dataset_paths(img_list, dst_dir, need_test=False):
    random.seed(1)
    file_num = len(img_list)
    train_num = 811
    random.shuffle(img_list)
    train_paths = img_list[: train_num]
    val_paths = [i for i in img_list if i not in train_paths]
    
