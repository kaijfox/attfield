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
    help = "Total number of stimuli to use in training each category's "
    "classifier. Remainder of images in that category's database in the HDF5 " +
    "archive will be used for validation - note the archive contains positive " +
    "and negative examples in each category database.") 
args = parser.parse_args()




# Setup
# ========================================================


# Process arguments / load data
embeddings_file = args.encodings
layer = args.layer
embed_h5 = h5py.File(embeddings_file, 'r')
n_train = args.n_train

# Create containers for coefs and scores for each category
all_coefs = []
all_apool_coefs = []
val_aucs = {'apool': [], 'fullmap': []}

# Note size of feature maps
map_size = embed_h5[layer].shape[4]
n_maps = embed_h5[layer].shape[3]




# Training
# ========================================================

# Train classifiers for each category
for i_cat in range(embed_h5[layer].shape[0]):
    print(f"Category {i_cat}")

    # Set up feature variables for this category
    feats = embed_h5[layer][i_cat, :n_train]
    # map_size = feats.shape[2]
    # n_maps = feats.shape[1]
    feats = feats.reshape([n_train, -1])


    # Train regressions on the full feature map before average pooling
    reg_args = dict(
        solver = 'liblinear',
        max_iter = 1000,
        fit_intercept = False)
    reg = LogisticRegression(**reg_args)
    ys = embed_h5['y'][i_cat, :n_train].astype('bool')
    reg.fit(feats, ys)

    # Set up validation set
    val_feats = embed_h5[layer][i_cat, n_train:]
    val_feats = val_feats.reshape([val_feats.shape[0], -1]) 
    val_ys = embed_h5['y'][i_cat, n_train:].astype('bool')
    
    # Measure performance on the validation set
    val_fn_scores = (reg.coef_ * val_feats).sum(axis = 1)
    val_auc = skmtr.roc_auc_score(val_ys, val_fn_scores)
    val_aucs['fullmap'].append(val_auc)

    # Save coefficients from this regression
    coefs = reg.coef_[0].reshape([n_maps, -1])
    all_coefs.append(coefs)


    # Train corresponding logistic regression on average pooled features
    apool_trn_feats = (feats
        ).reshape([n_train, n_maps, map_size, map_size]
        ).mean(axis = (2, 3))
    reg = LogisticRegression(**reg_args)
    reg.fit(apool_trn_feats, ys)

    # Measure validation performance
    apool_val_feats = (val_feats
        ).reshape([len(val_feats), n_maps, map_size, map_size]
        ).mean(axis = (2, 3))
    apool_val_fn = (reg.coef_ * apool_val_feats).sum(axis = 1)
    val_auc = skmtr.roc_auc_score(val_ys, apool_val_fn)
    val_aucs['apool'].append(val_auc)

    # Save resulting coefficients for this category
    apool_coefs = reg.coef_[0]
    all_apool_coefs.append(apool_coefs)



np.savez(args.output_path,
    fullmap_coefs = all_coefs,
    apool_coefs = all_apool_coefs,
    fullmap_auc = val_aucs['fullmap'],
    apool_auc = val_aucs['apool'],
    n_maps = n_maps,
    map_size = map_size)
