import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from lib.experiment.spatial_fields import TreeField, LinearField
from lib.experiment import voxel_selection as vx

from argparse import ArgumentParser
from scipy import spatial
from scipy import ndimage
import pandas as pd
import numpy as np



parser = ArgumentParser(
    description = 
        " summaries of receptive field statistics.")
parser.add_argument('output_path',
    help = 'Where to store the saved interpolator.')
parser.add_argument("focl",
    help = 'Focal condition (shifted) receptive fields.')
parser.add_argument('dist',
    help = 'Distributed condition receptive fields.')
parser.add_argument('layer',
    help = 'Layer whose units the field will be estimated from.')
parser.add_argument('norm', type = float,
    help = 'Size of input space.')
args = parser.parse_args()
args.layer = eval(args.layer)



# -------------------------------------- Load inputs ----

# Receptive fields
summ = pd.read_csv(args.focl)
units = vx.VoxelIndex.from_serial(summ['unit'])
summ.set_index('unit', inplace = True)

# Normalization RFs
center = pd.read_csv(args.dist)
center.set_index('unit', inplace = True)
if not all(center.index == summ.index):
    raise ValueError(
        f"Unit sets differ in {args.focl} and {args.dist}")


# ----------------------------- Compute location & shift ----

# Only look at units from the given layer
lstr = '.'.join(str(i) for i in args.layer)
mask = summ.index.map(lambda u: u.startswith(lstr))

# Calculate stat
shift_r = (summ.loc[mask, 'com_y'] - center.loc[mask, 'com_y']).values
shift_r /= args.norm
shift_c = (summ.loc[mask, 'com_x'] - center.loc[mask, 'com_x']).values
shift_c /= args.norm

# Build spatial map of input receptive fields
center = summ.loc[mask, ['com_y', 'com_x']].values.copy()
center /= args.norm
tree = spatial.KDTree(center)


# ----------------------------- Build interpolator object ----

t_field = TreeField(tree, shift_r, shift_c)
cs_grid, rs_grid = np.meshgrid( # Extend beyond [0,1] extent
    np.linspace(-0.1, 1.1, 50),
    np.linspace(-0.1, 1.1, 50))
grid_rshift, grid_cshift = t_field.query(rs_grid, cs_grid)

# Smooth
smooth_rshift = ndimage.gaussian_filter(grid_rshift, 1.5, mode = 'nearest')
smooth_cshift = ndimage.gaussian_filter(grid_cshift, 1.5, mode = 'nearest')


field = LinearField(rs_grid, cs_grid, smooth_rshift, smooth_cshift)
field.save(args.output_path)






