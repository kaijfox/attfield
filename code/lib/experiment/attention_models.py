"""

Implentation of helper functions for loading attention models or other dynamically
specified scripts, and for parsing configurations to be passed to those scripts.

### Functions
- `load_model` : Load a global variable from a python file given the path.
- `load_cfg` : Load configs from JSON or a key-value string format.

"""


from lib.experiment import network_manager as nm
from lib.experiment import detection_task as det
from lib.experiment import deformed_conv as dfc
from lib.experiment import lsq_fields

import importlib.util
from torch import nn
import numpy as np
import torch
import json
import os



def load_model(filename, **kwargs):
    """
    Load the global variable `model` from filename, the implication being that
    the object is a `network_manager.LayerMod` to use as an attention model.
    """
    # Run the model file
    fname_hash = hash(filename)
    spec = importlib.util.spec_from_file_location(
        f"model_file_{fname_hash}", filename)
    model_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_file)
    return model_file.attn_model(**kwargs)


def load_cfg(string):
    """
    Either load the JSON file pointed to by `string` or parse it as configs
    (`:`-sepearated statements of KEY=VALUE, where `VALUE` is parsed as a
    Python expression or string if evaluation fails, for example
    `"layer=(0,1,0):beta=4.0"` would become {'layer':(0,4,0), 'beta': 4.0})
    """
    if string is None:
        return {}
    elif string.endswith('.json'):
        with open(string) as f:
            return json.load(f)
    else:
        def try_eval(s):
            try: return eval(s)
            except: return s
        return {
                try_eval(s.split('=')[0].strip()): # LHS = key
                try_eval(s.split('=')[1].strip())  # RHS = val
            for s in string.split(':')
        }


def pct_to_shape(pcts, shape):
    """Scale percentage-based coordinate according to feature map shape.
    ### Arguments
    - `pcts` --- Pecentage-based coordinate, shape (ndim,)
    - `shape` --- Shape of feature map spatial dimensions, shape (ndim,)
    """
    return tuple(p * s for p, s in zip(pcts, shape[-len(pcts):]))




