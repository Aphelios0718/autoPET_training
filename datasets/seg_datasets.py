import json
import numpy as np
from datasets.utils_for_dataset import *
from monai.data.dataset import Dataset
from monai.transforms import apply_transform
from monai.data.image_reader import NumpyReader
from datasets.seg_transform import *


def read_splits_json(splits_json, phase="train"):
    f = open(splits_json, "r")
    data_list = json.load(f)[phase]
    assert os.path.exists(data_list[0])
    return data_list


class PETCT_seg(Dataset):
    def __init__(self, phase: str, args, reader_keys: list = None):

        self.phase = phase
        self.patch_size = args.patch_size
        self.data_reader = NumpyReader(reader_keys)
        print(f"reading data list from {args.data_root}/splits_final.json")
        self.data = read_splits_json(
            f"{args.data_root}/splits_final.json",
            phase=phase,
        )

        print(f"num of {self.phase} dataset: {len(self.data)}")

    def _transform(self, data):
        suv, ct = data[0].astype(np.float32), data[1].astype(np.float32)
        seg = data[2].astype(np.float32)
        _data = {"suv": suv, "ct": ct, "seg": seg}
        return apply_transform(get_transform(self.patch_size, self.phase), _data)

    def __getitem__(self, index):
        fname = self.data[index]
        _data = self.data_reader.read(fname)
        _data_ts = self._transform(data=_data)
        if isinstance(_data_ts, list):
            new_data_ts = []
            for item in _data_ts:
                new_data_ts.append({"fname": fname, **item})
            return new_data_ts

        else:
            return {"fname": fname, **_data_ts}

class PETCT_infer(Dataset):
    def __init__(self, data_list):
        
        self.data_list = data_list
        self.patch_size = (128, 128, 128)
        self.reader_keys = ["suv", "ct", "seg"]
        self.data_reader = NumpyReader(self.reader_keys)
        print(f"num of current dataset: {len(self.data_list)}")

    def _transform(self, data):
        _data = {}
        for k,v in zip(self.reader_keys, data):
            _data[k] = v
        return apply_transform(get_transform(self.patch_size, self.phase), _data)

    def __getitem__(self, index):
        fname = self.data[index]
        _data = self.data_reader.read(fname)
        _data_ts = self._transform(data=_data)
        
        if isinstance(_data_ts, list):
            new_data_ts = []
            for item in _data_ts:
                new_data_ts.append({"fname": fname, **item})
            return new_data_ts
        else:
            return {"fname": fname, **_data_ts}