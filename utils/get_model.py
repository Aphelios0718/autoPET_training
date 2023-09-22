import os
import torch
from models import colean_all
from monai.networks.nets import UNet


def resume_training(model, resume_path):
    assert os.path.isfile(resume_path)
    ori_model = torch.load(resume_path)
    print("-------- resume model from {} --------\n".format(resume_path))
    return model.load_state_dict(ori_model.state_dict())


def generate_model(opt):
    model = None

    if opt.model == "unet":
        from monai.networks.layers.factories import Norm

        model = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            num_res_units=2,
            channels=[32, 32, 64, 128, 256, 320],
            strides=[1, 2, 2, 2, 2],
            norm=Norm.INSTANCE,
        )

    if opt.model == "co_learn_unet":
        model = colean_all.CoLearnUNet(1, opt.n_seg_classes, opt.model)
        
    if opt.resume_path != "":
        resume_training(model, opt.resume_path)
    return model, model.parameters()
