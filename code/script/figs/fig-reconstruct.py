from lib.plot import behavior
from lib.plot import readouts
from lib.plot import kwargs as pkws
from lib.plot import util
from lib import paths

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

class params:
    # --- Aesthetics
    total_size = (pkws.onecol, 2*pkws.onecol) #cm

    # --- Figure and data outputs
    output = 'plots/figures/fig2/fig-reconstruct.pdf'
    cis_file = paths.data("ci_cmd.txt")

    # --- Data inputs
    acts_dist = paths.data('reconst/fnenc_task_base.h5')
    acts_focl = paths.data('reconst/enc_task_gauss_b4.0.h5')
    regs = paths.data('models/logregs_iso224_t100.npz') # classifiers.md


# ----------------  load data  ----

# load encoding/readout data
readout_data = readouts.readout_data(
    params.acts_dist, params.acts_focl,
    (0, 4, 3))
regs = readouts.load_logregs(params.regs)


# ----------------  make structure  ----

import matplotlib
sns.set('notebook')
sns.set_style('ticks')
matplotlib.rcParams.update(pkws.rc)

# make figure
cm = 1/2.54
fig = plt.figure(
    constrained_layout = False,
    figsize = [s*cm for s in params.total_size])

# make gridspec
gs = gridspec.GridSpec(
    nrows = 2, ncols = 1, figure = fig,
    height_ratios = [3, 2],
    **pkws.onecol_gridspec)
fine_gc =  gridspec.GridSpec(
    nrows = 6, ncols = 3, figure = fig,
    wspace = 0.3, hspace = 0.3,
    left = 0.1, right = 0.9,
    bottom = 0.06, top = 0.97)
base_a = util.panel_label(fig, fine_gc[0, 0], "a", xoffset = 0.1)
base_b = util.panel_label(fig, fine_gc[0, 2], "b", xoffset = 0.1)
base_c = util.panel_label(fig, gs[1, 0], "c", xoffset = 0.1)


# ----------------  gain map panel  ----

# panel a
ax_b = fig.add_subplot(fine_gc[0, 2])
gain_mappable = readouts.gain_map(ax_b, readout_data)
util.labels(ax_b, "Layer 4 feature map", "")
util.colorbar(
    fig, ax_b, gain_mappable,
    label = pkws.labels.effective_gain_short,
    shrink = 0,)


# ----------------  behavior panel  ----

# panel b
ax_c = fig.add_subplot(gs[1, 0])
bhv_cis = readouts.reconstructed_bhv(ax_c, readout_data, regs)
sns.despine(ax = ax_c, offset = 5)
util.labels(ax_c, None, pkws.labels.bhv_performance)
behavior.update_ci_text(params.cis_file,
    ReconstMultPerformance = behavior.group_ci_text(bhv_cis, 'dist_fake', '', col = 'cond', col_suffix = '_b'),
    ReconstMultFX = behavior.group_ci_text(bhv_cis, 'dist_fake', 'fx_', col = 'cond'),
    ReconstFakeDrop = behavior.group_ci_text(bhv_cis, 'focl_fake', 'fx_', col = 'cond'),
    ReconstUndoPerformance = behavior.group_ci_text(bhv_cis, 'dist_undo', '', col = 'cond', col_suffix = '_b'),
    ReconstUndoFX = behavior.group_ci_text(bhv_cis, 'dist_undo', 'fx_', col = 'cond'),)

# save
plt.savefig(params.output, transparent = True)











