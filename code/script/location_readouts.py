from lib.plot import readouts

from argparse import ArgumentParser
import sklearn.metrics as skmtr
import numpy as np

parser = ArgumentParser(
    description = 
        "Compute scores for neccessary/sufficient analysis on last-layer " + 
        "encodings using a masked readout over the target quadrant.")
parser.add_argument("output_path",
    help = "Path to .npz file to save the resulting data in.")
parser.add_argument("coef_file",
    help = "Classifier coefficient file trained by train_fullmap_logregs.py")
parser.add_argument("dist_file",
    help = "Path to HDF5 archive containing of last-layer encodings of " +
    "stimuli from the detection task in the *distributed* condition.")
parser.add_argument("focl_file",
    help = "Path to HDF5 archive containing of last-layer encodings of " +
    "stimuli from the detection task in the *focal* condition.") 
args = parser.parse_args()


# Load data
# ========================================================

coef_data = np.load(args.coef_file)
fullmap_readout_data = readouts.readout_data(
    args.dist_file, args.focl_file,
    (0, 4, 3))


# Supporting functions
# ========================================================

def per_pix_auc(pos, neg):
    """Compute AUC between samples of confidence scores in positive and
    negative ground truth conditions across spatial dimensions."""
    ytrue = np.concatenate([
        np.ones(pos.shape[0]),
        np.zeros(neg.shape[0])])
    scores = np.zeros([7, 7])
    for r in range(7):
        for c in range(7):
            fn = np.concatenate([
                pos[..., r, c].sum(axis = -1),
                neg[..., r, c].sum(axis = -1)])
            scores[r, c] = skmtr.roc_auc_score(ytrue, fn)
    return scores


# Compute AUCs
# ========================================================

scores_dict = readouts.reconstructed_bhv_auc(
    fullmap_readout_data,
    coef_data['apool_coefs'].reshape([20, 512, 1, 1]),
    score_func = per_pix_auc)
np.savez(args.output_path, **scores_dict)