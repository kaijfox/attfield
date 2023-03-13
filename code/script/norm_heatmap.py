

from lib.experiment import voxel_selection as vx
from lib.plot.util import mean_ci, binned_mean_line


import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser
import numpy as np
import tqdm
import h5py
import os


parser = ArgumentParser(
    description = 
        "Train logistic regressions on isolated object detection task.")
parser.add_argument('output_path',
    help = 'Path to PDF file where the plot should go.')
parser.add_argument("pre_acts",
    help = 'Encodings from each layer to be plotted without attention.')
parser.add_argument("post_acts", nargs = '+',
    help = 'One or more encoding files with attention applied.')
parser.add_argument("--loc", nargs = 2, type = float, required = True,
    help = 'Position of the center of the attention field (pct_x, pct_y)')
parser.add_argument("--rad", type = float, default = 1.,
    help = 'Radius of the attention field. Default 1, units = percent.')
parser.add_argument('--disp', nargs = '+', default = None, type = str,
    help = 'Display names for the post_acts files.')
parser.add_argument('--degrees_in_img', type = float, default = None,
    help = 'Ratio of degrees of visual angle to image size. If not given ' +
           'then shifts displayed in percentages of image size.',)
parser.add_argument('--exponent', default = [1,2,3,4], type = float, nargs = '+'
    help = "List of exponents to use in nonlinear normalization.")
parser.add_argument('--raw_ylim', default = (None, None), nargs = 2, type = float,
    help = "Y-axis limits for the raw activation magnitude plot")
parser.add_argument( '--sd_ylim', default = (None, None), nargs = 2, type = float,
    help = "Y-axis limits for the activation std-dev plot")
parser.add_argument('--n_img', default = None, type = int,
    help = "Number of images to include in the std-dev computation.")
parser.add_argument('--n_feat', default = float('inf'), type = int,
    help = "Number of feature maps on which to compute gain.")
parser.add_argument('--normalize', default = None, type = float, nargs = 2,
    help = "Apply feature-map-wise nonlinear normalization: floating point" + 
        "constants: exponent, sigma.")
parser.add_argument('--figsize', default = (6,6), nargs = 2, type = float)
parser.add_argument("--no_read", action = 'store_false',
    help = "Do not use cached '.sd.npz' or '.sgain.npz' files; recompute gain for" +
        "all inputs.")
parser.add_argument('--layers', default = None, nargs = '+', type = str,
    help = "Subset of layers in the provided encoding files to compute gain for.")
parser.add_argument('--loc_field', default = None, type = float, nargs = 2,
    help = "Percentage of the image field to consider 'locus' in gain calculation")
parser.add_argument('--n_bins', default = 7, type = int)
parser.add_argument('--bootstrap_n', default = 1000, type = int,
    help = 'Number of iterations in boostrap statistics.')
parser.add_argument('--no_raw', action = 'store_true',
    help = 'Do not plot raw activations')
parser.add_argument('--no_line', action = 'store_true',
    help = 'Do not plot LOESS line')
parser.add_argument('--is_comparison', action = 'store_true')
args = parser.parse_args()





"""
Test args:
class args:
    output_path = '/tmp/tmp.pdf'
    pre_acts = 'data/runs/fig2/lenc_task_base.h5'
    post_acts = [
        'data/runs/fig2/lenc_task_gauss_b2.0.h5',
        'data/runs/fig2/lenc_task_gauss_b4.0.h5']
    # post_acts = [
    #     'data/runs/fig2/lenc_task_gauss_b2.0.h5']
    loc = [.25, .25]
    rad = .25
    disp = ["2.0", "4.0"]
    degrees_in_img = 1
    norm_summ = 'data/runs/210420/summ_base_soft.csv'
    norm_param = 'rad'
    pal_f = 'data/cfg/pal_beta.csv'
    pal_l = 'data/cfg/pal_beta.csv'
    figsize = (6, 6)
    raw_ylim = (0.6, 1.2e5)
    sd_ylim = (0.6, 6)
    n_img = 1
"""

# -------------------------------------- Load inputs ----

# Distributed activations
# shape of acts[i][layer]: (cat, img, feat, row, col)
pre_acts = h5py.File(args.pre_acts, 'r+')
layers = [
    l for l in pre_acts.keys()
    if args.layers is None or l in args.layers]

# Focal activations
post_acts = []
for fname in args.post_acts:
    post_acts.append(h5py.File(fname, 'r+'))
    if not all([l in post_acts[-1] for l in layers]):
        raise ValueError(
            f"Layers sampled in {fname} do not match {args.pre_acts}")

# Get number of images and features at each layer
n_img = {
    l: (pre_acts[l].shape[1] if args.n_img is None else
        min(args.n_img, pre_acts[l].shape[1]))
    for l in layers}
n_feat = {
    l: min(pre_acts[l].shape[2], args.n_feat)
    for l in layers
}


# -------------------------- Supporting functions ----

# optional normalization
exponents = args.exponent #[1,2,3,4]
def normalized(feat_map):
    """
    Apply normalization according to args.normalize and args.exponent

    Parameters
    ----
    feat_map, np.ndarray, shape (category, img, feature, col, row)
        Feature map to normalize
    """
    results = []
    for EXPONENT in exponents:
        axis = (-3, -2, -1)
        norm_slice = lambda arr: arr[..., None, None, None]
        exp = feat_map ** EXPONENT
        normalizer = abs(exp).mean(axis = axis)
        normalizer /= abs(feat_map).mean(axis = axis)
        results.append(exp / norm_slice(normalizer))
    return np.stack(results)


def locus_field_ratio(dists, gains):
    if args.loc_field is None: return None
    locus_mask = dists <= np.quantile(dists, args.loc_field[0])
    field_mask = dists >= np.quantile(dists, args.loc_field[1])
    return gains[locus_mask].mean() / gains[field_mask].mean()


# -------------------------- Precompute location & shift ----
# Note: .norm.sgain.npz files are the main contribution of this script, as
# plotting of gains is done in each figure file based on these cached data


# Calculate distance from center in units of radii
dists = {}
ws = {}
hs = {}
for l in layers:
    ws[l] = pre_acts[l].shape[-1]
    hs[l] = pre_acts[l].shape[-2]
    cs, rs = np.meshgrid(np.linspace(0, 1, ws[l]), np.linspace(0, 1, hs[l]))
    dists[l] = np.sqrt(((cs - args.loc[0]) * args.degrees_in_img) ** 2 +
                       ((rs - args.loc[1]) * args.degrees_in_img) ** 2)




# calculate standard-deviation based gain
# Load cached data if exists and allowed
if os.path.exists(args.pre_acts + '.norm.sd.npz') and args.no_read:
    print("Loading stddev ", args.pre_acts + '.norm.sd.npz')
    pre_sds = {k: v for k, v in np.load(args.pre_acts + '.norm.sd.npz').items()}
else:
    # Calculate actvation standard deviation for each unit and cache results
    pre_sds = {}
    for l in layers:
        feat_sds = [
                normalized(pre_acts[l][:, :n_img[l], i_feat]).std(axis = (1,2))
                for i_feat in tqdm.trange(n_feat[l], position = 0)]
        pre_sds[l] = np.stack(feat_sds)
    print("Saved normalized base stddev to ", args.pre_acts + '.norm.sd.npz')
    np.savez(args.pre_acts + '.norm.sd.npz', **pre_sds)


sd_gains = []
lfs = []
zero_div = lambda a,b: np.divide(a, b, out = np.zeros_like(a), where = b!=0)
for i_f in range(len(post_acts)):
    # Load cached data if exists and allowed
    if os.path.exists(args.post_acts[i_f] + '.norm.sgain.npz') and args.no_read:
        print("Loading SD gains ", args.post_acts[i_f] + '.norm.sgain.npz')
        sd_gains.append({k: v for k, v in np.load(args.post_acts[i_f] + '.norm.sgain.npz').items()})
    else:
        # Calculate standard deviation ratio (effective gain) for each unit and cache
        sd_gains.append({})
        for l in layers:
            feat_sds = [
                zero_div(
                    normalized(
                        post_acts[i_f][l][:, :n_img[l], i_feat]
                    ).std(axis = (1,2)),
                    pre_sds[l][i_feat])
                for i_feat in tqdm.trange(n_feat[l], position = 0)]
            sd_gains[i_f][l] = np.stack(feat_sds, axis = 1)

        print("Saved normalized sd gains to", args.post_acts[i_f] + '.norm.sgain.npz')
        np.savez(args.post_acts[i_f] + '.norm.sgain.npz', **sd_gains[i_f])

    # Identify locus-field ratio for plotting
    lfs.append({})
    for l in layers:
        lfs[i_f][l] = {}
        for i_exp in range(len(exponents)):
            xs = np.tile(dists[l][None, :, :], [n_feat[l], 1, 1]).ravel()
            ys = sd_gains[i_f][l][i_exp].ravel()
            if not args.no_line:
                plt.figure()
                plt.plot(xs, ys, 'o')
                plt.show()
                plt.close()
        lfs[i_f][l][i_exp] = locus_field_ratio(xs, ys)



# -------------------------------------- Plot ----

for l in layers:
    fig, ax = plt.subplots()
    sns.heatmap(
        np.array([[lfs[i_f][l][i_exp]
                   for i_exp in range(len(exponents))]
                   for i_f in range(len(post_acts))]),
        ax = ax)
    ax.set_title(l)
    ax.set_yticks(np.arange(len(args.disp))[::-1] + 0.5)
    ax.set_xticks(np.arange(len(exponents)) + 0.5)
    ax.set_yticklabels(args.disp, va = 'center')
    ax.set_xticklabels(exponents)
    plt.savefig(args.output_path)
    plt.show()








