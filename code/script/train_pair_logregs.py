from sklearn.linear_model import LogisticRegression
from argparse import ArgumentParser
import sklearn.metrics as skmtr
import numpy as np
import h5py



parser = ArgumentParser(
    description = 
        "Train logistic regressions on full output layer feature maps (before " + 
        "average pooling) as well as on features after average pooling.")
parser.add_argument("output_path",
    help = "Path to .npz file to save the resulting data.")
parser.add_argument("encodings",
    help = "Path to HDF5 archive containing of last-layer encodings of " +
    "stimuli from positive and negative isolated images of each class.")
parser.add_argument("layer",
    help = "Layer to pull activations from in the encodings file, ex. '0.4.3'")
parser.add_argument("n_train", type=int,
    help = "Total number of stimuli to use from each of positive and negative " + 
    "categories in training each classifier.")
parser.add_argument("n_val", type=int,
    help = "Total number of stimuli to use from each of positive and negative " + 
    "categories in testing each classifier.") 
args = parser.parse_args()




# Setup
# ========================================================

iso_h5 = h5py.File(args.encodings, 'r')
layer = args.layer
n_trn_each = args.n_train
n_val_each = args.n_val

# Determine category pairs
# Alternating categories provides standardized pairs here and in 
# other paired-category experiments, since categories in isolated
# archive always have same order
cats = [c for c in iso_h5.keys() if not c.endswith('_y')]
pair_c1 = np.arange(0, (len(cats) // 2) * 2, 2)
pair_c2 = pair_c1 + 1

# Containers for data during iteration
regs = {}
val_aucs = {}



# Training
# ========================================================

# For each category pair, train and test classifier 
for i_pair, (c1, c2) in enumerate(zip(pair_c1, pair_c2)):
    print(f"Category: pair{i_pair}")

    # Pull positive examples from category 1 for training and validation
    c1_y = iso_h5['y'][c1].astype('bool')
    trn_c1_feat = iso_h5[layer][c1][...][c1_y][:n_trn_each]
    val_c1_feat = iso_h5[layer][c1][...][c1_y][n_trn_each:n_trn_each + n_val_each]

    # Pull positive examples from category 2 for training and validation
    c2_y = iso_h5['y'][c2].astype('bool')
    trn_c2_feat = iso_h5[layer][c2][...][c2_y][:n_trn_each]
    val_c2_feat = iso_h5[layer][c2][...][c2_y][n_trn_each:n_trn_each + n_val_each]

    # Construct X,Y datasets for fitting
    trn_x = np.hstack([trn_c1_feat, trn_c2_feat]
        ).reshape((n_trn_each*2,) + trn_c1_feat.shape[1:]
        ).mean(axis = (2, 3))
    val_x = np.hstack([val_c1_feat, val_c2_feat]
        ).reshape((n_val_each*2,) + val_c1_feat.shape[1:]
        ).mean(axis = (2, 3))
    trn_y = np.array([1, 0] * n_trn_each).astype('bool')
    val_y = np.array([1, 0] * n_val_each).astype('bool')

    # Train regression, test, and save coefficients
    reg = LogisticRegression(
        solver = 'liblinear',
        max_iter = 1000,
        fit_intercept = False)
    reg.fit(trn_x, trn_y)
    trn_fn = (reg.coef_ * trn_x).sum(axis = 1)
    val_fn = (reg.coef_ * val_x).sum(axis = 1)
    regs[f'pair{i_pair}'] = reg.coef_



regs = {**regs, **{f'{k}_auc':v for k,v in val_aucs.items()}}
np.savez(args.output_path, **regs)