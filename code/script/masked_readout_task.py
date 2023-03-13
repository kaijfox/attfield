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
    "stimuli from the detection task in the distributed condition.")
parser.add_argument("focl_file",
    help = "Path to HDF5 archive containing of last-layer encodings of " +
    "stimuli from the detection task in the focal condition.") 
parser.add_argument("mask_size", type=int,
    help = "Side length of the square at the top left corner (overlapping "
    "mostly the target quadrant) to include in readout.") 
args = parser.parse_args()




# Load data
# ========================================================

coef_data = np.load(args.coef_file)
fullmap_readout_data = readouts.readout_data(
    args.dist_file, args.focl_file,
    (0, 4, 3))


# masked readout reconstruct plot (data only)
# =========================================================

def mask_auc(mask):
    """Generate a scoring function for `reconstructed_bhv_auc` that applies
    a multiplicative mask to the feature map before calculating performance."""
    def score_func(pos, neg):
        ytrue = np.concatenate([
            np.ones(pos.shape[0]),
            np.zeros(neg.shape[0])])

        fn = np.concatenate([
            (mask[None, None] * pos).mean(axis = (-2, -1)).sum(axis = -1),
            (mask[None, None] * neg).mean(axis = (-2, -1)).sum(axis = -1)])
        return skmtr.roc_auc_score(ytrue, fn)
    return score_func


# generate readout mask
mask = np.zeros([7, 7])
mask[:args.mask_size, :args.mask_size] = 1

# compute scored in dist, focal, multiplied, divided conditions
# using the readout mask
scores_dict = readouts.reconstructed_bhv_auc(fullmap_readout_data,
    coef_data['apool_coefs'].reshape([20, 512, 1, 1]),
    score_func = mask_auc(mask))

# save resulting scores
np.savez(args.output_path, **scores_dict)