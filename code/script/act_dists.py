"""
TODO
    - what is n_bins
    - what is_comparison"""

"""
Plot effective gain changes between a distributed ('pre') and various focal
('post') conditions using precomputed activations.

Most importantly, outputs cached HDF5 archives containing effective gain
measurements: for each focal activation archive data/FOCAL.h5 given, an archive
data/FOCAL.h5.sgain.npz will be generated, with keys for each layer in the
activations file mapping to an array of effective gains with shape 
`(n_features, n_col, n_row)`
"""


from lib.experiment import voxel_selection as vx
from lib.plot.util import mean_ci, binned_mean_line
from lib.plot import kwargs as pkws

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser
import pandas as pd
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
    help = 'Position of the center of the attention field in percent of' + 
        'input space, i.e. --loc 0.5 0.5 for center of stimulus.')
parser.add_argument("--rad", type = float, default = 1.,
    help = 'Radius of the attention field. Default 1, units = percent.')
parser.add_argument('--disp', nargs = '+', default = None, type = str,
    help = 'Display names for the post_acts files.')
parser.add_argument('--pal_f', default = None, type = str,
    help = 'Color palette to use when color spread across the different' + 
           'post-attention activation files.')
parser.add_argument('--pal_l', default = None, type = str,
    help = 'Color palette to use when color spread across the different' + 
           'layers of the network.')
parser.add_argument('--degrees_in_img', type = float, default = None,
    help = 'Ratio of degrees of visual angle to image size. If not given ' +
           'then shifts displayed in percentages of image size.',)
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
parser.add_argument('--layernorm', default = None, type = float, nargs = 2,
    help = "Apply layer-wise nonlinear normalization: floating point" + 
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


# -------------------------------------- Load inputs ----

# Distributed activations
# shape of acts[i][layer]: (cat, img, feat, row, col)
pre_acts = h5py.File(args.pre_acts, 'r+')
layers = [
    l for l in pre_acts.keys()
    if (args.layers is None or l in args.layers) and (l != 'y')]
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

# Color palettes for plotting
if args.pal_f is not None: pal_f = pd.read_csv(args.pal_f)['color']
else: pal_f = ['#0288D1', '#C62828', '#FFB300', '#5E35B1', '#43A047']

if args.pal_l is not None: pal_l = pd.read_csv(args.pal_l)['color']
else: pal_l = ['#0288D1', '#C62828', '#FFB300', '#5E35B1', '#43A047']


# -------------------------- Supporting functions ----

# optional normalization
def normalized(feat_map):
    """
    Apply normalization according to args.normalize and args.layernorm
    Parameters
    ----
    feat_map, np.ndarray, shape (category, img, feature, col, row)
        Feature map to normalize
    """
    if args.normalize is None and args.layernorm is None:
        return feat_map
    if args.normalize is not None:
        SIGMA, EXPONENT = args.normalize
        axis = (-2, -1)
        norm_slice = lambda arr: arr[..., None, None]
    elif args.layernorm is not None:
        SIGMA, EXPONENT = args.layernorm
        axis = (-3, -2, -1)
        norm_slice = lambda arr: arr[..., None, None, None]
    exp = feat_map ** EXPONENT
    normalizer = abs(exp).mean(axis = axis) + SIGMA ** EXPONENT
    normalizer /= abs(feat_map).mean(axis = axis)
    return exp / norm_slice(normalizer)


def locus_field_ratio(dists, gains):
    """
    Determine ratio of gain between locus and remaining image field.

    Parameters
    ----
    dists, np.ndarray, shape (n_units,)
        Distances of units from the locus of attention
    gains, np.ndarray, shape (n_units,)
        Gain for each unit
    """
    if args.loc_field is None: return None
    locus_mask = dists <= np.quantile(dists, args.loc_field[0])
    field_mask = dists >= np.quantile(dists, args.loc_field[1])
    return gains[locus_mask].mean() / gains[field_mask].mean()


# -------------------------- Precompute location & shift ----
# Note: .sgain.npz files are the main contribution of this script, as
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
if os.path.exists(args.pre_acts + '.sd.npz') and args.no_read:
    print("Loading stddev ", args.pre_acts + '.sd.npz')
    pre_sds = {k: v for k, v in np.load(args.pre_acts + '.sd.npz').items()}
else:
    # Calculate actvation standard deviation for each unit and cache
    pre_sds = {}
    for l in layers:
        feat_sds = [
                normalized(pre_acts[l][:, :n_img[l], i_feat]).std(axis = (0,1))
                for i_feat in tqdm.trange(n_feat[l], position = 0)]
        pre_sds[l] = np.stack(feat_sds)
    print("Saved base stddev to ", args.pre_acts + '.sd.npz')
    np.savez(args.pre_acts + '.sd.npz', **pre_sds)

sd_gains = []
zero_div = lambda a,b: np.divide(a, b, out = np.zeros_like(a), where = b!=0)
for i_f in range(len(post_acts)):
    # Load cached data if exists and allowed
    if os.path.exists(args.post_acts[i_f] + '.sgain.npz') and args.no_read:
        print("Loading SD gains ", args.post_acts[i_f] + '.sgain.npz')
        sd_gains.append({k: v for k, v in np.load(args.post_acts[i_f] + '.sgain.npz').items()})

    else:
        # Calculate standard deviation ratio (effective gain) for each unit and cache
        sd_gains.append({})
        for l in layers:
            print(n_img[l])
            feat_sds = [
                zero_div(
                    normalized(
                        post_acts[i_f][l][:, :n_img[l], i_feat]
                    ).std(axis = (0,1)),
                    pre_sds[l][i_feat])
                for i_feat in tqdm.trange(n_feat[l], position = 0)]
            sd_gains[i_f][l] = np.stack(feat_sds)

        print("Saved sd gains to", args.post_acts[i_f] + '.sgain.npz')
        np.savez(args.post_acts[i_f] + '.sgain.npz', **sd_gains[i_f])



# -------------------------------------- Plot ----
# Note: This plotting code kept as further example of using effective gain
# measurments, but does not contribute to any analyses in the paper


sns.set('notebook')
sns.set_style('ticks')
with PdfPages(args.output_path) as pdf:

    gain_arrs = (
        [sd_gains, "Change in activation std. [fraction]",
         'linear', args.sd_ylim],
    )

    # Plot axis = layer, color = beta
    for l in layers:
        print(f"Plot: axis-layer, color-beta; layer {l}")

        for gain_arr, ylab, yscl, ylim in gain_arrs:

            fig, ax = plt.subplots(figsize = args.figsize)
            for i_f in range(len(post_acts)):
                xs = np.tile(dists[l][None, :, :], [n_feat[l], 1, 1]).ravel()
                ys = gain_arr[i_f][l].ravel()
                lf = locus_field_ratio(xs, ys)
                if not args.no_raw:
                    ax.plot(xs, ys,
                        ms = 1, marker = 'o', ls = '', color = pal_f[i_f],
                        alpha = 0.7, zorder = 1,
                        label = (
                            (args.disp[i_f] if args.disp is not None else None) + 
                            (f", l/f={lf:.4f}" if lf is not None else "")),
                        rasterized = True)

                if not args.no_line:
                    bin_centers, bin_means, low_ci, high_ci = binned_mean_line(
                        xs, ys, args.n_bins, args.bootstrap_n)
                    if args.is_comparison: line_kws = plot.kwargs.errorbar_secondary
                    else: line_kws = plot.kwargs.errorbar
                    ax.errorbar(
                        bin_centers, bin_means,
                        (bin_means - low_ci, high_ci - bin_means),
                        **line_kws, zorder = 2,
                        color = pal_f[i_f])
            plt.yscale(yscl)
            

            if args.disp is not None:
                ax.legend(frameon = False)

            ax.set_ylabel(ylab)
            ax.set_title(
                f'Layer: {l}' +
                f" | Locus/Field: {lf:.4f}" if lf is not None else "")
            ax.set_xlabel("Unit distance from attention locus [%]")
            ax.set_ylim(*ylim)

            sns.despine(ax = ax)
            plt.tight_layout()
            pdf.savefig(transparent = True)
            plt.close()
       
    # Plot with axis = beta, color = layer
    # Todo: all beta on one axis if there's no difference across layers
    for i_f in range(len(post_acts)):
        print(f"Plot: axis-beta, color-layer; file",
              args.disp[i_f] if args.disp is not None else None)

        for gain_arr, ylab, yscl, ylim in gain_arrs:

            fig, ax = plt.subplots(figsize = args.figsize)
            for i_l, l in enumerate(layers):
                xs = np.tile(dists[l][None, :, :], [n_feat[l], 1, 1]).ravel()
                ys = gain_arr[i_f][l].ravel()
                lf = locus_field_ratio(xs, ys)
                if not args.no_raw:
                    ax.plot(xs, ys,
                        ms = 1, marker = 'o', ls = '', color = pal_l[i_l],
                        alpha = 0.7,  zorder = 1, rasterized = True,
                        label = f"Layer: {l}" + 
                                f", l/f={lf:.4f}" if lf is not None else "")

                if not args.no_line:
                    bin_centers, bin_means, low_ci, high_ci = binned_mean_line(
                        xs, ys, args.n_bins, args.bootstrap_n)
                    if args.is_comparison: line_kws = plot.kwargs.errorbar_secondary
                    else: line_kws = plot.kwargs.errorbar
                    ax.errorbar(
                        bin_centers, bin_means,
                        (bin_means - low_ci, high_ci - bin_means),
                        **line_kws, zorder = 2,
                        color = pal_l[i_l])
            plt.yscale(yscl)

            

            if args.disp is not None:
                ax.legend(frameon = False)

            ax.set_ylabel(ylab)
            ax.set_title(
                args.disp[i_f] if args.disp is not None else None)
            ax.set_xlabel("Unit distance from attention locus [%]")
            ax.set_ylim(*ylim)

            sns.despine(ax = ax)
            plt.tight_layout()
            pdf.savefig(transparent = True)
            plt.close()

    # Plot each layer, beta on different axis
    for gain_arr, ylab, yscl, ylim in gain_arrs:
        for i_f in range(len(post_acts)):
        
            for l in layers:
                print(f"Separate axes; file={i_f}, layer={l}")
                xs = np.tile(dists[l][None, :, :], [n_feat[l], 1, 1]).ravel()
                ys = gain_arr[i_f][l].ravel()

                fig, ax = plt.subplots(figsize = args.figsize)
                if not args.no_raw:
                    ax.plot(xs, ys,
                        ms = 1, marker = 'o', ls = '', color = '.3',
                        alpha = 0.7, zorder = 1, rasterized = True)

                if not args.no_line:
                    bin_centers, bin_means, low_ci, high_ci = binned_mean_line(
                        xs, ys, args.n_bins, args.bootstrap_n)
                    if args.is_comparison: line_kws = plot.kwargs.errorbar_secondary
                    else: line_kws = plot.kwargs.errorbar
                    ax.errorbar(
                        bin_centers, bin_means,
                        (bin_means - low_ci, high_ci - bin_means),
                        **line_kws, zorder = 2,
                        color = '.2')
                plt.yscale(yscl)

                lf = locus_field_ratio(
                    np.tile(dists[l][None, :, :], [n_feat[l], 1, 1]).ravel(),
                    gain_arr[i_f][l].ravel())

                ax.set_ylabel(ylab)
                ax.set_xlabel("Unit distance from attention locus [%]")
                ax.set_title(
                    f"Layer: {l} | " + 
                    (args.disp[i_f] if args.disp is not None else None) +
                    (f" | Locus/Field: {lf:.4f}" if lf is not None else ""))
                ax.set_xlabel("Unit distance from attention locus [%]")
                ax.set_ylim(*ylim)

                sns.despine(ax = ax)
                plt.tight_layout()
                pdf.savefig(transparent = True)
                plt.close()






