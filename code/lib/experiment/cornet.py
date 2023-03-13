"""

Utility function for loading CORnet models.

"""


import os.path
import torch

from lib import paths


def load_cornet(model_type,
    model = None,
    code_root = paths.code.join('cornet/cornet'),
    weight_root = paths.data.join("models")):

    # If model not given then load one according the given cornet type name
    if model is None:
        if model_type == 'Z':
            model = os.path.join(code_root, "cornet_z.py")
        elif model_type == 'S':
            model = os.path.join(code_root, "cornet_s.py")

    ckpt_data = torch.load(
        os.path.join(weight_root, "cornet_" + model_type.upper() + '.pth.tar'),
        map_location=lambda storage, loc: storage)
    model.load_state_dict({
        '.'.join(k.split(".")[1:]): v
        for k,v in ckpt_data['state_dict'].items()})

    return model, ckpt_data