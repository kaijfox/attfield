"""

Implementation of the Guassian gain attention model, as used in `fig-cornet`,
`fig-gauss`, `fig-reconstruct`, and supplements. Applies a gain field with 
peak `beta` before the targeted layer.

"""


from lib.experiment import network_manager as nm

import numpy as np
import torch

def pct_to_shape(pcts, shape):
    """Scale percentage-based coordinate according to feature map shape.
    ### Arguments
    - `pcts` --- Pecentage-based coordinate, shape (ndim,)
    - `shape` --- Shape of feature map spatial dimensions, shape (ndim,)
    """
    return tuple(p * s for p, s in zip(pcts, shape[-len(pcts):]))

class GaussianLocatedGain(nm.LayerMod):
    def __init__(self, center, r, beta):
        '''
        ### Arguments
        - `center` --- Center location of the gaussian field in the input,
            a tuple of the form (row_center, col_center)
        - `r` --- Approximate radius of influence of the gaussian field,
            a tuple of the form (row_r, col_r)
        - `beta` --- Multiplicative strength factor
        '''
        super(GaussianLocatedGain, self).__init__()
        self.center = center
        self.r = r
        self.beta = beta

    def pre_layer(self, inp, *args, **kwargs):
        """
        ### Arguments
        - `inp` --- Main layer input, of shape (batch, channel, row, col)
        """
        scaled = inp * self.scale_array(inp)
        return (scaled,) + args, kwargs, None

    def scale_array(self, match):
        """Compute gain map to match a given input tensor"""
        shape = match.shape
        # interpolate percentage-unit params up to layer scale
        loc = pct_to_shape(self.center, shape)
        rad = pct_to_shape(self.r, shape)
        # Create grid
        r = np.broadcast_to(np.arange(shape[-2])[:, None], shape[-2:])
        c = np.broadcast_to(np.arange(shape[-1])[None, :], shape[-2:])
        # Gaussian field
        local_r = (r - loc[0]) / rad[0]
        local_c = (c - loc[1]) / rad[1]
        G = np.exp( - local_r**2 / 2 ) * np.exp( - local_c**2 / 2 )
        G = (self.beta - 1) * G + 1
        # Match dtype and device of input tensor
        return torch.tensor(G, dtype = match.dtype, device = match.device)


def attn_model(layer, beta, loc = (0.25, 0.25), r = (0.125, 0.125)):
    return  {
        tuple(layer): GaussianLocatedGain(loc, r, beta)
    }