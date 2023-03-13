"""

Implementation of the sensitivity-shift model used in `fig-sensitivity`.
Applies a gain field to the weights of a Conv2D layer and renormalizes,
so that sensitivity is shifted but overall magnitude of weights remains
constant across the convolution.

"""


from lib.experiment import network_manager as nm
from lib.experiment import deformed_conv as dfc

from torch import nn
import torch
import numpy as np


def pct_to_shape(pcts, shape):
    return tuple(p * s for p, s in zip(pcts, shape[-len(pcts):]))


class NormalizedSensitivityGradAttention(nm.LayerMod):

    def __init__(self, center, r, beta):
        '''
        ### Arguments
        - `center` --- Center location of the gaussian field in the input,
            a tuple of the form (row_center, col_center)
        - `r` --- Approximate radius of influence of the gaussian field,
            a tuple of the form (row_r, col_r)
        - `beta` --- Multiplicative strength factor
        '''
        super(NormalizedSensitivityGradAttention, self).__init__()
        self.center = center
        self.r = r
        self.beta = beta
        self.filter_cache = {}

    def scale_array(self, match):
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
        # Match characteristic of input tensor
        return torch.tensor(G, dtype = match.dtype, device = match.device)

    def pre_layer(self, inp, *args, **kwargs):
        """
        ### Arguments
        - `inp` --- Main layer input, of shape (batch, channel, row, col)
        """
        scale_array = self.scale_array(inp)

        conv = kwargs['__layer']
        if not isinstance(conv, nn.Conv2d):
            raise NotImplementedError("NormalizedSensitivityGradAttention only" + 
                " implemented for wapping torch 2d convolutions. Was asked" + 
                " to wrap {}".format(type(conv)))

        # Set up mimicry of the layer we're wrapping
        if (conv, inp.shape) not in self.filter_cache:
            self.filter_cache = {}

            pad = dfc.conv_pad(conv)
            flt = dfc.broadcast_filter(conv)
            sten = dfc.filter_stencil(conv)
            grid = dfc.conv_grid(conv, inp.shape[2], inp.shape[2])
            ix = dfc.merge_grid_and_stencil(grid, sten)

            loc = pct_to_shape(self.center, inp.shape)
            rad = pct_to_shape(self.r, inp.shape)

            # Shift receptive fields
            # The factor 2 * (27 / 112) matches det.QuadAttention and scales
            # the gaussian field to have approximate radius sd.
            field =  dfc.make_gaussian_sensitivity_field(*loc,
                4 * rad[0] * (27 / 112), 4 * rad[1] * (27 / 112))
            gained_flt, normalizer = dfc.apply_magnitude_field(
                flt, ix, field, pad, amp = self.beta)

            self.filter_cache[(conv, inp.shape)] = normalizer
        else:
            normalizer = self.filter_cache[(conv, inp.shape)]

        if conv.bias is not None:
            bias_correction = conv.bias[:, None, None] * (normalizer - 1)

        return (inp * scale_array,) + args, kwargs, (normalizer, bias_correction)

    def post_layer(self, outputs, cache):
        '''Implement layer bypass, replacing the layer's computation
        with the deformed convolution'''
        # shape: (C_out, rows, cols)
        normalizer, bias_correction = cache
        norm_ret = outputs * normalizer[None] - bias_correction[None]
        return norm_ret




def attn_model(layer, beta, r = 0.25, **kws):
    '''
    - `neg_mode` --- True for warning, `'raise'` for exception, `'fix'` to offset
        feild locations with a negative to be 0 or positive.
    '''
    # One layer
    if isinstance(layer[0], int):
        return {
            tuple(layer): NormalizedSensitivityGradAttention((0.25, 0.25), (r, r), beta)
        }
    # Multiple layers
    else:
        return {
            tuple(L): NormalizedSensitivityGradAttention((0.25, 0.25), (r, r), beta)
            for L in layer
        }










def make_gaussian_sensitivity_field(mu_r, mu_c, sigma_r, sigma_c):
    def field(r, c):
        local_r = (r - mu_r) / sigma_r
        local_c = (c - mu_c) / sigma_c
        G = np.exp( - local_r**2 / 2 ) * np.exp( - local_c**2 / 2 )
        return G
    return field