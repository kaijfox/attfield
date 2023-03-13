"""

Convolutions with shifted filter grids and altered weights at each grid
location. For implementing the deformed convolution, the following terms
are helpful
- A `grid` defines the top left filter locations for a convolution, and
    is captured by an array with shape (`out_row`, `out_col`, 2)
- A `stencil` defines the offset of inputs to the convolutional filter
    from a given grid location. It is represented by an array with shape
    (`sten_row`, `sten_col`, 2)
- A `filter` is a set of weights applied by a convolution, and is
    given by an array of shape:
    (`channel_in`, `channels_out`, *, *, `sten_row`, `sten_col`)
    where "*, *" is either 1's or `out_row`, `out_col` if different
    weights should be applied at each filter location.


### Functions

- `conv_grid` - Construct the grid corresponding to a torch Conv2D
- `filter_stencil` - Construct the filter stencil corresponding to a
    torch Conv2D
- `conv_pad` - Extract padding information from a torch Conv2D
- `broadcast_filter` - Extract filter from a torch Conv2D
- `apply_magnitude_field` - Apply a gain field on the input space of a
    convolution to the filter array of the corresponding deformed
    convolution without altering weight norms at each grid location.    
- `merge_grid_and_stencil` - Convert a grid and stencil pair to absolute
    locations of each input to each grid location
- `merge_grid_and_nonuniform_stencil` - Equivalent to the function
    `merge_grid_and_stencil`, but accepts stencils that vary according
    to the grid location
- `take_conv_inputs` - Index into an input array as defined by a merged
    grid and stencil. Used directly in `deformed_conv`.
- `deformed_conv` - Apply a filter to an input according to a merged
    grid and stencil (apply deformed convolution).
- `rigid_shift` - Special case of deformed convolution in which the
    stencil and filter are not altered, so hardware accelerated
    convolution operation can still be used.


"""


from torch import nn
import numpy as np
import torch
import gc





# =======================================================================
# -  Manipulating grids, stencils, filters, etc                         -
# =======================================================================


def conv_grid(conv, in_rows, in_cols):
    '''
    For a given torch Conv2D and input shape, return the convolution
    grid before deformation.

    ### Returns
    - `grid` --- `np.ndarray` of shape (out_row, out_col, 2) giving
        the index in the input image corresponding to the top-left-most
        component of the filter
    '''

    if conv.groups != 1:
        raise NotImplementedError("conv_grid not implemented for " + 
                                  "grouped convolutions.")
    if conv.dilation != (1, 1):
        raise NotImplementedError("conv_grid not implemented for " + 
                                  "dilated convolutions.")

    # Note: https://github.com/vdumoulin/conv_arithmetic/
    # is a great resource for understanding these computations

    max_r_ix = (in_rows + 2 * conv.padding[0] 
                - (conv.kernel_size[0] - 1))
    r_ix = np.arange(0, max_r_ix, conv.stride[0])

    max_c_ix = (in_cols + 2 * conv.padding[1] 
                - (conv.kernel_size[1] - 1))
    c_ix = np.arange(0, max_c_ix, conv.stride[1])

    return np.stack(np.meshgrid(c_ix, r_ix)[::-1], axis = -1)




def filter_stencil(conv):
    '''
    Create a stencil to match the operation performed by the given Conv2d

    ### Arguments
    - `conv` --- `nn.Conv2d` layer to match.

    ### Returns
    - `stencil` --- `np.ndarray` shape
        (sten_row, sten_col, 2)
    '''

    if conv.groups != 1:
        raise NotImplementedError("conv_grid not implemented for " + 
                                  "grouped convolutions.")

    # Note: https://github.com/vdumoulin/conv_arithmetic/
    # is a great resource for understanding these computations

    r_ix = np.arange(conv.kernel_size[0]) * conv.dilation[0]
    c_ix = np.arange(conv.kernel_size[1]) * conv.dilation[1]
    return np.stack(np.meshgrid(c_ix, r_ix)[::-1], axis = -1)


def conv_pad(conv):
    '''
    Extract padding information from a Conv2D to use for `take_conv_inputs`
    '''
    return (conv.padding[1], conv.padding[1], conv.padding[0], conv.padding[0])


def broadcast_filter(conv):
    '''
    Create a filter tensor applying the same operatios as `conv` would.

    ### Arguments
    - `conv` --- `nn.Conv2d` layer to match.

    ### Returns
    - `flt` --- Torch parameter.
        Shape (C_out, C_in, 1, 1, sten_rows, sten_cols), notably broadcasts
        with (C_out, C_in, out_row, out_col, sten_rows, sten_cols) and
        so is compatible with `deformed_conv`
    '''
    return conv.weight[:, :, None, None, :, :]








def apply_magnitude_field(flt, ix, field_fn, pad, amp):
    """
    ### Arguments
    - `flt` --- Torch tensor, elongated filter, shape
        shape: (C_out, C_in, out_row, out_col, sten_rows, sten_cols)
        Note that these dimensions can also be `1` in which case
        torch/numpy will broadcast appropriately.
    - `ix` --- Numpy array of locations which the filter applies to,
        shape: (out_row, out_col, sten_row, sten_col, 2)
    - `field_fn` --- A function taking two arguments: (row, col)
        and which returns the field value at the point. Should
        permit vectorization by passing row, col as vectors.
    - `amp` --- Multiplicative strength factor of the field adjustment
    """

    # Shape: (out_row * out_col * sten_row * sten_col)
    field = field_fn(ix[..., 0].ravel() - pad[2], ix[..., 1].ravel() - pad[0])
    # Shape: (1, 1, out_row, out_col, sten_row, sten_col)
    field = field.reshape(ix.shape[:-1])
    field = (field * (amp-1)) + 1
    

    # Apply multiplicative factors and return
    field = torch.tensor(field[None, None, ...], dtype = flt.dtype)
    if flt.is_cuda:
        field = field.cuda()
    # shape: (C_out, C_in, out_row, out_col, sten_rows, sten_cols)
    scaled_flt = field * flt

    # calculate new magnitudes to normalize by
    representative_flt_new = scaled_flt
    # shape: (C_out, out_row, out_col)
    flt_magnitudes_new = torch.sqrt((representative_flt_new ** 2).sum(dim = (1, -2, -1)))
    # assume filter magnitudes were the same per-channel before applying field
    # shape: (C_out, C_in, 1, 1, sten_rows, sten_cols)
    representative_flts_old = flt[:, :, 0, 0, :, :][:, :, None, None, :, :]
    # shape: (C_out, 1, 1)
    flt_magnitudes_old = torch.sqrt((representative_flts_old ** 2).sum(dim = (1, -2, -1)))
    # shape: (C_out, out_row, out_col)
    normalizer = flt_magnitudes_old / flt_magnitudes_new
    renormed = normalizer[:, None, :, :, None, None] * scaled_flt

    return renormed, normalizer
    




# =======================================================================
# -  Perform operations using grids, stencils, etc                      -
# =======================================================================



def merge_grid_and_stencil(conv_grid, filter_stencil):
    '''
    Combine a grid and stencil to get convolution indices
    ### Returns
    - `ix` --- `np.ndarray` shape (out_row, out_col, sten_row, sten_col, 2)
    '''

    # Shape: (out_row, out_col, 2, 1, 1)
    grid_broadcast = conv_grid[..., None, None]
    # Shape: (1, 1, 2, filter_rows, filter_cols)
    filter_stencil = np.moveaxis(filter_stencil, -1, 0)
    sten_broadcast = filter_stencil[None, None, ...]

    ix = grid_broadcast + sten_broadcast
    ix = np.moveaxis(ix, 2, -1)

    return ix



def merge_grid_and_nonuniform_stencil(conv_grid, filter_stencil):
    '''
    Combine a grid and stencil to get convolution indices, but with
    a stencil that varies according to output postion
    ### Parameters
    - `grid` --- `array` shape (out_row, out_col, 2)
    - `stencil` --- `array` shape (out_row, out_col, sten_row, sten_col, 2)
    ### Returns
    - `ix` --- `np.ndarray` shape (out_row, out_col, sten_row, sten_col, 2)
    '''

    # Shape: (out_row, out_col, 1, 1, 2,)
    grid_broadcast = conv_grid[:, :, None, None, :]
    # Shape: (out_row, out_col, filter_rows, filter_cols, 2)
    sten_broadcast = filter_stencil

    ix = grid_broadcast + sten_broadcast

    return ix



def take_conv_inputs(inp, ix, pad):
    '''
    Get a view of the input tensor (N, C, H, W) as indexed by
    applying the given stencil at every location of the given grid.
    
    ### Returns
    - `sliced` --- `torch.tensor`
        shape: (N, C, out_row, out_col, sten_row, sten_col)
    '''

    # To understand what this is doing:
    #inp = torch.arange(50).view(2, 5, 5)
    #inp[..., ((1, 2), (1, 2)), ((3, 4), (3, 4))].shape
    #r_ix = np.array(((1, 2), (1, 2), (1, 2)))
    #c_ix = np.array(((1, 2), (1, 2), (1, 2)))
    #torch.arange(50).view(2, 5, 5)[..., r_ix, c_ix]


    # Correct for indices outside the input image
    r_tp_pad = -min(0, ix[..., 0].min())
    r_bt_pad = max(0, ix[..., 0].max() - (inp.shape[2]-1))
    c_lf_pad = -min(0, ix[..., 1].min())
    c_ri_pad = max(0, ix[..., 1].max() - (inp.shape[3]-1))
    pad = (pad[0] + c_lf_pad, pad[1] + c_ri_pad,
           pad[2] + r_tp_pad, pad[3] + r_bt_pad)
    r_ix = ix[..., 0] + r_tp_pad
    c_ix = ix[..., 1] + c_lf_pad

    if not all(p == 0 for p in pad):
        inp = nn.functional.pad(inp, pad = pad, mode = 'constant', value = 0.)
    ret = inp[..., r_ix, c_ix]
    del r_ix, c_ix;
    gc.collect()
    return ret




def deformed_conv(inp, ix, flt, pad, bias = None):
    '''
    ### Arguments
    - `flt` --- Elongated filter to apply, for example, the
        output of `broadcast_filter`. Shape must be
        (C_out, C_in, out_row, out_col, sten_rows, sten_cols)
        Note that these dimensions can also be `1` in which case
        torch will broadcast appropriately.
    '''


    # Perform the convolution at 4 (integer) reference locations to 
    # interpolate between

    op_keys = {0: np.floor, 1: np.ceil}
    ref_convs = {}

    # Add batch dimension to filter
    # shape: (1, C_out, C_in, out_row, out_col, sten_rows, sten_cols)
    flt = flt[None, ...]

    for row_op in range(2):
        for col_op in range(2):

            # Apply floor or ceiling to get real indexes from floats
            rounded_ix = np.stack((
                op_keys[row_op](ix[..., 0]),
                op_keys[col_op](ix[..., 1])
            ), axis = -1).astype('long')

            # Elongate input
            # shape: (N, C_out, C_in, out_row, out_col, sten_rows, sten_cols)
            inp_view = take_conv_inputs(inp, rounded_ix, pad)
            inp_view = inp_view[:, None, ...]

            # Perform convolution
            # shape: (N, C_out, C_in, out_row, out_col, sten_rows, sten_cols)
            torch.cuda.empty_cache()
            conved = inp_view * flt
            

            # shape: (N, C_out, out_row, out_col, sten_rows, sten_cols)
            conved = torch.sum(conved, dim = 2)
            ref_convs[(row_op, col_op)] = conved

    # Perform bilinear interpolation weighted by the float indices

    # Get fractional part of indices (interpolation weights)
    c = np.remainder(ix[..., 0].astype('float'), 1)
    r = np.remainder(ix[..., 1].astype('float'), 1)

    # Shape: (1, 1, out_row, out_col, sten_row, sten_col)
    c = torch.tensor(c, dtype = inp.dtype, device = ref_convs[0,0].device)[None, None, ...]
    r = torch.tensor(r, dtype = inp.dtype, device = ref_convs[0,0].device)[None, None, ...]

    # shape: (N, C_out, out_row, out_col, sten_row, sten_col)
    conved = (
        (1 - c) * (1 - r) * ref_convs[0, 0] +
             c  * (1 - r) * ref_convs[1, 0] + 
        (1 - c) *      r  * ref_convs[0, 1] + 
             c  *      r  * ref_convs[1, 1])


    # Perform final sum of convolution and add bias
    # Shape: (N, C_out, out_row, out_col, sten_row, sten_col)
    conved = torch.sum(conved, dim = (-2, -1))
    if bias is not None:
        conved = conved + bias[None, :, None, None]

    return conved



def rigid_shift(inputs, grid):
    """
    Index into the last two dimensions of `inputs` by `grid`
    with bilinear interpolation.
    """

    # don't leave image bounds
    grid[..., 0] = np.clip(grid[..., 0], 0, inputs.shape[-2] - 1)
    grid[..., 1] = np.clip(grid[..., 1], 0, inputs.shape[-1] - 1)

    # weights
    c = np.remainder(grid[..., 0].astype('float'), 1)
    r = np.remainder(grid[..., 1].astype('float'), 1)
    c = torch.tensor(c, dtype = inputs.dtype, device = inputs.device)
    r = torch.tensor(r, dtype = inputs.dtype, device = inputs.device)

    op_keys = {0: np.floor, 1: np.ceil}
    ref_convs = {}
    weights = {
        ('r', 0): (1 - c),
        ('r', 1): c,
        ('c', 0): (1 - r),
        ('c', 1): r
    }

    shifted = None

    for row_op in range(2):
        for col_op in range(2):

            # Apply floor or ceiling to get real indexes from floats
            rounded_ix = np.stack((
                op_keys[row_op](grid[..., 0]),
                op_keys[col_op](grid[..., 1])
            ), axis = -1).astype('long')

            contrib = inputs[..., rounded_ix[..., 0], rounded_ix[..., 1]]
            contrib *= weights['r', row_op] * weights['c', col_op]

            if shifted is None:
                shifted = contrib
            else:
                shifted += contrib

    return shifted



    


