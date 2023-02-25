"""
Run a model on a set of inputs and save its activations.
This should usually be done on small sets of images, or the output
files will become prohibitively large.

Main outputs are an HDF5 archive of activations, containing datasets 
  - For each observed layer, i.e. `(0,4,1)` (see network_manager.py for
    info on layer specifications), a dataset named for the 'layer string',
    i.e. `'0.4.1'` with shape (n_categories, n_imgs, n_features, ... rest
    of tensor shape) where the output tensor is taken to have shape
    (n_imgs, n_features, ...). Note `n_imgs` and `n_features` are treated
    separately due to possible limit of number of features with `--max_feat`.
    If a layer outputs a tuple of tensors, then the corresponding datasets will
    be named `'0.4.1_0'`, `'0.4.1_1'`, etc. 
  - A dataset "y" of shape (n_categories, n_imgs) containing ground truth labels.
If linear network heads are provided, an additional HDF5 archive will be 
created, with `fn_` prepended to the filename, containing datasets named for
each observed layer (or each tensor output from the layer in the case of a 
tuple as above) having shape (n_categories, n_imgs, n_regressions, ... rest of 
tensor shape), with the regression's decision function applied along the 
feature axis of the tensor (axis 1).
"""


from lib.experiment import detection_task as det
from lib.experiment import attention_models as atts
from lib.experiment import network_manager as nm
from lib.experiment import cornet


from argparse import ArgumentParser
import importlib.util
import numpy as np
import h5py
import os


parser = ArgumentParser(
    description = 
        "Extract full layer activations.")
parser.add_argument('output_path',
    help = 'Path to an HDF5 file where activations should be stored.')
parser.add_argument("image_gen",
    help = 'Path to a python script generating inputs.')
parser.add_argument('model', type = str, default = None,
    help = 'Python file with a function `get_model` that returns a PyTorch'+
           'model for the script to run backprop on. If not provided, the '+
           'script will use CorNet-Z.')
parser.add_argument("layers", nargs = "+",
    help = 'List of layers to pull encodings of as tuples, i.e. "(0,4,1)"')
parser.add_argument("--attn", default = None,
    help = 'Path to a python file defining attention to apply. The '+
           '`attn_model` function defined in the file will be called '+
           'to instantiate a LayerMod implementing attention.')
parser.add_argument('--attn_cfg', default = None,
    help = 'Path to a JSON file to pass as kwargs to `attn_model()` '+
           'from the attention model file.')
parser.add_argument('--gen_cfg', default = None,
    help = 'Path to a JSON file to pass as kwargs to `generate_set()` '+
           'from the image gen file.')
parser.add_argument('--regs', default = None,
    help = 'Regression objects to run on each spatial location.')
parser.add_argument('--batch_size', type = int, default = -1,
    help = 'If given data will be run in batches.')
parser.add_argument('--max_feat', type = int, default = float('inf'),
    help = 'Max number of feautres to output from each layer.')
parser.add_argument('--cuda', action = 'store_true',
    help = 'Force data and weight tensors to reside on GPU.')
args = parser.parse_args()
args.layers = [eval('tuple('+l+')') for l in args.layers]


# -------------------------------------- Load inputs ----

# Neural network observer model
if args.model is None:
    model, _ = cornet.load_cornet("Z")
else:
    spec = importlib.util.spec_from_file_location(
        "model", args.model)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model = model_module.get_model()
# Image generator config
gen_kws = atts.load_cfg(args.gen_cfg)
# Image generator
spec = importlib.util.spec_from_file_location(
    "image_gen", args.image_gen)
image_gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_gen_module)
generate_set = image_gen_module.generate_set
# Attention model
if args.attn is not None:
    kws = atts.load_cfg(args.attn_cfg)
    att_mods = atts.load_model(args.attn, **kws)
else:
    att_mods = {}
# Linear network heads
if args.regs is not None:
    regs = det.load_logregs(args.regs, bias = False)

# And prep output object
outputs = None
if args.regs is not None:
    fn_outputs = None


# ------------------------------------------------- Run model ----

# Generate images/stimuli
meta, imgs = generate_set(**gen_kws)

# Helper function for treating layer outputs
def suffixes_and_tensors(computed_layer):
    """Convert between layers that return tensors and tuples of tensors"""
    if isinstance(computed_layer, tuple):
        suffixes = [f'_{tns_ix}' for tns_ix in range(len(computed_layer))]
        layer_tensors = computed_layer
    else:
        suffixes = ('',)
        layer_tensors = (computed_layer,)
    return suffixes, layer_tensors

# Iterate over groups from the stimulus generator (categories)
for i_grp, ((grp_key, grp_meta), grp_imgs) in enumerate(zip(meta, imgs())):
    print("Group:", grp_key)

    # Organize the stimuli into batches
    batches = (np.arange(len(grp_imgs)), )
    if args.batch_size > 0:
        if len(grp_imgs) > args.batch_size:
            n_batches = len(grp_imgs)//args.batch_size
            batches = np.array_split(batches[0], n_batches)

    # Iterate over batches
    for batch_n, batch_ix in enumerate(batches):
        print(f"Batch {batch_n+1} / {len(batches)}")

        # Run the neural network observer on the stimuli
        batch_imgs = grp_imgs[batch_ix]
        mgr = nm.NetworkManager.assemble(model, batch_imgs,
            mods = att_mods, with_grad = False, cuda = args.cuda)

        # Generate the output HDF5 archive on first pass, now that layer
        # sizes are observed
        if outputs is None:
            outputs = h5py.File(args.output_path, 'w')

            # Create an HDF5 dataset for each tensor output by each observed layer
            for layer in args.layers:
                lstr = '.'.join(str(l) for l in layer)
                for suff, tns in zip(*suffixes_and_tensors(mgr.computed[layer])):
                    n_feat = min(tns.shape[1], args.max_feat)
                    outputs.create_dataset(lstr + suff,
                        (len(meta), len(grp_imgs), n_feat) + tns.shape[2:],
                        np.float32)
            
            # Store ground truth ouputs for each stimulus
            outputs.create_dataset('y', (len(meta), len(grp_imgs)), np.uint8)

            # Set up datasets for outputs of regressions
            if args.regs is not None:
                fn_outputs = h5py.File(
                    os.path.dirname(args.output_path) + 
                    '/fn_' + os.path.basename(args.output_path), 'w')
                for layer in args.layers:
                    lstr = '.'.join(str(l) for l in layer)
                    for suff, tns in zip(*suffixes_and_tensors(mgr.computed[layer])):
                        fn_outputs.create_dataset(lstr + suff,
                            (len(meta), len(grp_imgs), len(regs)) + tns.shape[2:],
                            np.float32)

        # Save ground truth for the category
        outputs['y'][i_grp] = grp_meta['ys']

        # Save activations for each layer
        for layer in args.layers:
            lstr = '.'.join(str(l) for l in layer)

            # Handle layers that produce a tuple of tensors
            for suff, tns in zip(*suffixes_and_tensors(mgr.computed[layer])):
                # Push activations to the archive
                enc = tns.detach().cpu()
                n_feat = min(enc.shape[1], args.max_feat)
                outputs[lstr+suff][i_grp, batch_ix.tolist()] = enc[:, :n_feat]

                # If linear network heads were passed, run their decision
                # function along axis 1 (feature map axis) of the tensor
                # and save to the `fn_`-prepended archive.
                if args.regs is not None:
                    for i_c, c in enumerate(regs):
                        fns = np.apply_along_axis(
                            regs[c].decision_function,
                            1, enc
                        ).squeeze()
                        fn_outputs[lstr+suff][i_grp, batch_ix.tolist(), i_c] = fns

outputs.close()
if args.regs is not None:
    fn_outputs.close()











