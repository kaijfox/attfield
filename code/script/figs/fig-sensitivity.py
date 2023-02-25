from lib.plot import lineplots
from lib.plot import quivers
from lib.plot import behavior
from lib.plot import kwargs as pkws
from lib.plot import util
from lib import paths

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import os

class params:
    # --- Aesthetics
    total_size = pkws.twocol_size #cm

    # --- Figure and data outputs
    output = 'plots/figures/fig5/fig-sensitivity.pdf'
    cis_file = paths.data("ci_cmd.txt")

    # --- Data inputs
    # Receptive fields
    dist_ells = paths.data('cornet/summ_base_ell.csv') # fig-cornet.md
    focl_ells = [
        paths.data('sensitivity/ell_sn4_n100_b1.1_ell.csv'),
        paths.data('sensitivity/ell_sn4_n100_b2.0_ell.csv'),
        paths.data('sensitivity/ell_sn4_n100_b4.0_ell.csv'),
        paths.data('sensitivity/ell_sn4_n100_b11.0_ell.csv'),
    ]
    # Receptive fields: gaussian comparison
    comp_ells = [
        paths.data('cornet/summ_cts_gauss_b1.1_ell.csv'), # fig-cornet.md
        paths.data('cornet/summ_cts_gauss_b2.0_ell.csv'), # fig-cornet.md
        paths.data('cornet/summ_cts_gauss_b4.0_ell.csv'), # fig-cornet.md
        paths.data('cornet/summ_cts_gauss_b11.0_ell.csv'), # fig-cornet.md
    ]
    # Behavior
    bhv_dist = paths.data('cornet/bhv_base.h5') # fig-cornet.md
    bhv_focl = {
        ('1.1', 'Gauss'): paths.data('cornet/bhv_gauss_n600_beta_1.1.h5'), # fig-cornet.md
        ('2.0', 'Gauss'): paths.data('cornet/bhv_gauss_n600_beta_2.0.h5'), # fig-cornet.md
        ('4.0', 'Gauss'): paths.data('cornet/bhv_gauss_n600_beta_4.0.h5'), # fig-cornet.md
        ('11.0', 'Gauss'): paths.data('cornet/bhv_gauss_n600_beta_11.0.h5'), # fig-cornet.md
        ('1.1', 'al'): paths.data('sensitivity/sna_bhv_n300_b1.1.h5'),
        ('2.0', 'al'): paths.data('sensitivity/sna_bhv_n300_b2.0.h5'),
        ('4.0', 'al'): paths.data('sensitivity/sna_bhv_n300_b4.0.h5'),
        ('11.0', 'al'): paths.data('sensitivity/sna_bhv_n300_b11.0.h5'),
        ('1.1', 'l1'): paths.data('sensitivity/sn1_bhv_n300_b1.1.h5'),
        ('2.0', 'l1'): paths.data('sensitivity/sn1_bhv_n300_b2.0.h5'),
        ('4.0', 'l1'): paths.data('sensitivity/sn1_bhv_n300_b4.0.h5'),
        ('11.0', 'l1'): paths.data('sensitivity/sn1_bhv_n300_b11.0.h5'),
        ('1.1', 'l2'): paths.data('sensitivity/sn2_bhv_n300_b1.1.h5'),
        ('2.0', 'l2'): paths.data('sensitivity/sn2_bhv_n300_b2.0.h5'),
        ('4.0', 'l2'): paths.data('sensitivity/sn2_bhv_n300_b4.0.h5'),
        ('11.0', 'l2'): paths.data('sensitivity/sn2_bhv_n300_b11.0.h5'),
        ('1.1', 'l3'): paths.data('sensitivity/sn3_bhv_n300_b1.1.h5'),
        ('2.0', 'l3'): paths.data('sensitivity/sn3_bhv_n300_b2.0.h5'),
        ('4.0', 'l3'): paths.data('sensitivity/sn3_bhv_n300_b4.0.h5'),
        ('11.0', 'l3'): paths.data('sensitivity/sn3_bhv_n300_b11.0.h5'),
        ('1.1', 'l4'): paths.data('sensitivity/sn4_bhv_n300_b1.1.h5'),
        ('2.0', 'l4'): paths.data('sensitivity/sn4_bhv_n300_b2.0.h5'),
        ('4.0', 'l4'): paths.data('sensitivity/sn4_bhv_n300_b4.0.h5'),
        ('11.0', 'l4'): paths.data('sensitivity/sn4_bhv_n300_b11.0.h5'),
    }
    # Effective gain
    sgain_focl = [
        paths.data('sensitivity/lenc_sna_n100_b1.1.h5.sgain.npz'),
        paths.data('sensitivity/lenc_sna_n100_b2.0.h5.sgain.npz'),
        paths.data('sensitivity/lenc_sna_n100_b4.0.h5.sgain.npz'),
        paths.data('sensitivity/lenc_sna_n100_b11.0.h5.sgain.npz'),
    ]
    # Effective gain: gaussian comparison
    sgain_comp = [
        paths.data('gauss/lenc_task_gauss_b1.1.h5.sgain.npz'), # fig-gauss.md
        paths.data('gauss/lenc_task_gauss_b2.0.h5.sgain.npz'), # fig-gauss.md
        paths.data('gauss/lenc_task_gauss_b4.0.h5.sgain.npz'), # fig-gauss.md
        paths.data('gauss/lenc_task_gauss_b11.0.h5.sgain.npz'), # fig-gauss.md
    ]

    # --- General parameters
    minigrids = False
    bhv_labels = ['All Layers', "Layer 1", "Layer 2", "Layer 3", "Layer 4"]
    size_lim = (0.85, 1.1)
    shift_lim = (-2.5, 15)
    gain_lim = (0, 10)




# ----------------  load data  ----

# load lineplot data
lp_pre_ells, lp_att_ells, lp_dists, lp_dists_px = lineplots.rf_data(
    params.dist_ells, params.focl_ells,
    loc = (56, 56), rad = 1)
_, lp_comp_ells, _, _ = lineplots.rf_data(
    params.dist_ells, params.comp_ells,
    loc = (56, 56), rad = 1)

# load behavior data
bhv_focl_data, bhv_dist_data = behavior.bhv_data(
    params.bhv_focl, params.bhv_dist, "Dist.")

# process sizemap/quiver data
qv_dist_ell, qv_focl_ell, qv_smooth_samp = quivers.quiver_data(
    lp_pre_ells, lp_att_ells[3], (0, 4, 0), 200)

# load gain data
sgain_focl = lineplots.gain_data(
    lp_pre_ells, params.sgain_focl, loc = (56, 56))
sgain_comp = lineplots.gain_data(
    lp_pre_ells, params.sgain_comp, loc = (56, 56))


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
    nrows = 3, ncols = 3, figure = fig,
    wspace = 0.4, hspace = 0.45,
    left = 0.06, top = 0.96, right = 0.96, bottom = 0.06)
base_a = util.panel_label(fig, gs[0, 0], "a")
base_b = util.panel_label(fig, gs[1, 0], "b")
base_c = util.panel_label(fig, gs[1, 1], "c")
base_d = util.panel_label(fig, gs[1, 2], "d")
base_e = util.panel_label(fig, gs[2, 0], "e")
base_f = util.panel_label(fig, gs[2, 1], "f")


# ----------------  middle row  ----

# panel b
ax_d = fig.add_subplot(gs[1, 0])
size_map = quivers.quiverplot(
    qv_dist_ell, qv_focl_ell, qv_smooth_samp,
    ax_d, cmap = 'coolwarm', vrng = params.size_lim)
util.axis_expand(ax_d, L = 0.2, B = 0.2, R = -0.1, T = 0.05)
util.labels(ax_d,
    pkws.labels.image_position.format('Horizontal'),
    pkws.labels.image_position.format('Vertical'))
util.colorbar(
    fig, ax_d, size_map, ticks = params.size_lim + (1,),
    label = pkws.labels.rf_size, label_vofs = -0.04)


# panel c : single axis
ax_c = fig.add_subplot(gs[1,1])
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'shift', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells),
    ax_c,
    line_span = 30, rad = 30, pal = pkws.pal_b,
    xlim = (0, 180), ylim = params.shift_lim)
util.labels(ax_c, pkws.labels.unit_distance, pkws.labels.rf_shift)
util.legend(
    fig, ax_c, 
    ['Gain strength'] + pkws.labels.beta,
    np.concatenate([[pkws.legend_header_color], pkws.pal_b.values]),
    inset = pkws.legend_inset)


# panel d : single axis
ax_d = fig.add_subplot(gs[1,2])
lineplots.lineplot(
    lineplots.rf_file_iterator(
        'size', lp_dists, lp_att_ells, (0,4,0),
        comp_ells = lp_comp_ells),
    ax_d,
    line_span = 30, rad = 30, pal = pkws.pal_b,
    xlim = (0, 180), ylim = params.size_lim)
util.labels(ax_d, pkws.labels.unit_distance, pkws.labels.rf_size)


# ----------------  bottom row  ----


#  panel e
gs_e = gs[2, 0].subgridspec(4, 2, **pkws.mini_gridspec,
    width_ratios = [2, 1])
# e: breakout
ax_e = np.array([fig.add_subplot(gs_e[i, 1]) for i in range(4)])
e_data_iter = list(lineplots.gain_file_iterator(
    lp_dists, sgain_focl, (0,4,0),
    gain_comp = sgain_comp))
lineplots.mini_lineplot(
    e_data_iter,
    ax_e.ravel(),
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.gain_lim)
# e: main panel
ax_e = fig.add_subplot(gs_e[:, 0])
lineplots.lineplot(
    e_data_iter,
    ax_e,
    line_span = pkws.lineplot_span, rad = 30, pal = pkws.pal_b,
    xlim = pkws.lineplot_xlim, ylim = params.gain_lim)
util.labels(ax_e, pkws.labels.unit_distance, pkws.labels.effective_gain)
# output confidence intervals on resulting gain
gain_mean_ci_table = util.mean_ci_table(
    [os.path.basename(f) for f in params.sgain_focl],
    [focl for _, _, focl, _ in e_data_iter],
    1000)
gain_sd_ci_table = util.mean_ci_table(
    [os.path.basename(f) for f in params.sgain_focl],
    [focl for _, _, focl, _ in e_data_iter],
    1000, aggfunc = lambda x: x.std(axis = 1))
behavior.update_ci_text(params.cis_file,
    SensGainMean = behavior.group_ci_text(gain_mean_ci_table, 'lenc_sna_n100_b4.0.h5.sgain.npz', ''),
    SensGainSD = behavior.group_ci_text(gain_sd_ci_table, 'lenc_sna_n100_b4.0.h5.sgain.npz', ''))


#  panel f
ax_f = fig.add_subplot(gs[2, 1:])
bhv_pal = np.concatenate([pkws.pal_bhv, pkws.pal_l.values])
bhv_cis = behavior.bhv_plot(
    bhv_focl_data, bhv_dist_data,
    bar1 = behavior.d2auc(0.75), bar2 = behavior.d2auc(1.28), dodge = 0.11,
    ax = ax_f, yrng = pkws.bhv_yrng, pal = bhv_pal,
    jitter = 0.02, bootstrap_n = 1000)
util.legend(fig, ax_f,
    [pkws.labels.gaussian_model, params.bhv_labels[0]],
    pkws.pal_bhv,
    inset = pkws.legend_inset, inset_y = pkws.legend_inset / 4,
    left = True)
util.legend(fig, ax_f,
    params.bhv_labels[1:3], pkws.pal_l.values[0:2],
    inset = 2.2, inset_y = pkws.legend_inset / 4,
    left = True)
util.legend(fig, ax_f,
    params.bhv_labels[3:], pkws.pal_l.values[2:4],
    inset = 3.5, inset_y = pkws.legend_inset / 4,
    left = True)
behavior.update_ci_text(params.cis_file,
    GaussPerformancePointOne = behavior.ci_text(bhv_cis, 'Gauss', '1.1', ''),
    GaussPerformanceTwo = behavior.ci_text(bhv_cis, 'Gauss', '2.0', ''),
    GaussPerformanceFour = behavior.ci_text(bhv_cis, 'Gauss', '4.0', ''),
    GaussPerformanceEleven = behavior.ci_text(bhv_cis, 'Gauss', '11.0', ''),
    SensPerformancePointOne = behavior.ci_text(bhv_cis, 'al', '1.1', ''),
    SensPerformanceTwo = behavior.ci_text(bhv_cis, 'al', '2.0', ''),
    SensPerformanceFour = behavior.ci_text(bhv_cis, 'al', '4.0', ''),
    SensPerformanceEleven = behavior.ci_text(bhv_cis, 'al', '11.0', ''),
    SensFXPointOne = behavior.ci_text(bhv_cis, 'al', '1.1', 'fx_'),
    SensFXTwo = behavior.ci_text(bhv_cis, 'al', '2.0', 'fx_'),
    SensFXFour = behavior.ci_text(bhv_cis, 'al', '4.0', 'fx_'),
    SensFXEleven = behavior.ci_text(bhv_cis, 'al', '11.0', 'fx_'))


# save
plt.savefig(params.output, transparent = True)











