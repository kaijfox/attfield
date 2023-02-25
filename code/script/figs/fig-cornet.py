import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from lib.plot import behavior
from lib.plot import kwargs as pkws
from lib.plot import util
from lib import paths

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

class params:
    # --- Aesthetics
    total_size = pkws.twocol_size #cm

    # --- Figure and data outputs
    output = 'plots/figures/fig2/fig-cornet.pdf'
    cis_file = paths.data('ci_cmd.txt')

    # --- Data inputs
    bhv_dist = paths.data('cornet/bhv_base.h5') # fig-cornet.md
    bhv_focl = {
        ('1.1', 'Gauss'): paths.data('cornet/bhv_gauss_n600_beta_1.1.h5'), # fig-cornet.md
        ('2.0', 'Gauss'): paths.data('cornet/bhv_gauss_n600_beta_2.0.h5'), # fig-cornet.md
        ('4.0', 'Gauss'): paths.data('cornet/bhv_gauss_n600_beta_4.0.h5'), # fig-cornet.md
        ('11.0', 'Gauss'): paths.data('cornet/bhv_gauss_n600_beta_11.0.h5'), # fig-cornet.md
    }


# ----------------  load data  ----

# load behavior data
bhv_focl_data, bhv_dist_data = behavior.bhv_data(
    params.bhv_focl, params.bhv_dist, "Dist.")


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
    nrows = 3, ncols = 2, figure = fig,
    **pkws.twocol_gridspec)
base_a = util.panel_label(fig, gs[0, 0], "a")
base_b = util.panel_label(fig, gs[0, 1], "b")
base_c = util.panel_label(fig, gs[1, 0], "c")
base_d = util.panel_label(fig, gs[2, 0], "d")

# ----------------  top row  ----

# panel a
ax_a = fig.add_subplot(gs[0, 0])
bhv_cis = behavior.bhv_plot(
    bhv_focl_data, bhv_dist_data,
    bar1 = behavior.d2auc(0.75), bar2 = behavior.d2auc(1.28),
    ax = ax_a, yrng = pkws.bhv_yrng, pal = pkws.pal_bhv,
    jitter = 0.03, bootstrap_n = 1000)
util.axis_expand(ax_a, L = -0.1, B = 0, R = 0, T = 0)
behavior.update_ci_text(params.cis_file,
    DistPerformance = behavior.ci_text(bhv_cis, "Dist.", "Dist.", ''),
    GaussPerformance = behavior.ci_text(bhv_cis, "Gauss", "4.0", ''),
    GaussFX = behavior.ci_text(bhv_cis, "Gauss", "4.0", 'fx_'))

# save
plt.savefig(params.output, transparent = True)











