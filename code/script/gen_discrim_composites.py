
from argparse import ArgumentParser
import numpy as np
import h5py



parser = ArgumentParser(
    description = 
        "Train logistic regressions on full output layer feature maps (before " + 
        "average pooling) as well as on features after average pooling.")
parser.add_argument("output_path",
    help = "Path to HDF5 archive in which to save composites.")
parser.add_argument("iso_imgs",
    help = "Path to HDF5 archive isolated images from each category.")
parser.add_argument("n_gen", type=int,
    help = "Number of 'category 1' composites and 'category 2' composites to " + 
    "generate for each pair.")
parser.add_argument("start_img", type=int,
    help = "No images from before this index in the isolated images archive" + 
    "will be used to generate composites - to avoid overlap with training.") 
args = parser.parse_args()




# Setup
# ========================================================



iso_h5 = h5py.File(args.iso_imgs, 'r')
n_gen = args.n_gen
start = args.start_img

# Determine category pairs
# Alternating categories provides standardized pairs here and in 
# other paired-category experiments, since categories in isolated
# archive always have same order
cats = [c for c in iso_h5.keys() if not c.endswith('_y')]
pair_c1 = np.arange(0, (len(cats) // 2) * 2, 2)
pair_c2 = pair_c1 + 1


# Generation
# ========================================================

with h5py.File(args.output_path, 'w') as out_h5:
    for i_pair, (c1, c2) in enumerate(zip(pair_c1, pair_c2)):
        print(f"Category: pair{i_pair}")

        # Get positive example images of each class
        c1_y = iso_h5[cats[c1] + '_y'].astype('bool')
        c1_imgs = iso_h5[cats[c1]][...][c1_y][start:start+n_gen]
        c2_y = iso_h5[cats[c1] + '_y'].astype('bool')
        c2_imgs = iso_h5[cats[c2]][...][c2_y][start:start+n_gen]

        # Choose indices of images to make up the composites
        d_ix = np.stack([
            np.random.choice(n_gen, [3, 2], replace = False)
            for _ in range(n_gen)], axis = 1)
        pd_ix = d_ix[:, :, 0]
        nd_ix = d_ix[:, :, 1]
        d_cls = np.random.choice(2, [3, n_gen], replace = True)
        nt_ix = np.random.permutation(n_gen)

        # Assemble composites
        imgs = np.stack([c1_imgs, c2_imgs])
        pos = np.concatenate([
            np.concatenate([                 c1_imgs, imgs[d_cls[1], pd_ix[1]]], axis = 2),
            np.concatenate([imgs[d_cls[0], pd_ix[0]], imgs[d_cls[2], pd_ix[2]]], axis = 2)
        ], axis = 1)
        neg = np.concatenate([
            np.concatenate([          c2_imgs[nt_ix], imgs[d_cls[1], nd_ix[1]]], axis = 2),
            np.concatenate([imgs[d_cls[0], nd_ix[0]], imgs[d_cls[2], nd_ix[2]]], axis = 2)
        ], axis = 1)

        # interleave positive and negative images
        all_imgs = np.hstack((pos, neg)).reshape(
            (n_gen * 2, c1_imgs.shape[1] * 2, c1_imgs.shape[2] * 2, c1_imgs.shape[3]))
        gen_ys = np.array([1, 0] * n_gen, dtype = np.bool_)

        dset_img = out_h5.create_dataset(f"pair{i_pair}", all_imgs.shape, np.float32)
        dset_img[...] = all_imgs
        dset_y = out_h5.create_dataset(f"pair{i_pair}_y", (n_gen * 2,), np.bool_)
        dset_y[...] = gen_ys
        dset_flp = out_h5.create_dataset(f"pair{i_pair}_cls.meta", (n_gen, 3), np.bool_)
        dset_flp[...] = d_cls.T