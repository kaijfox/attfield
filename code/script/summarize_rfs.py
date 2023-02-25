

from lib.experiment import voxel_selection as vx
from lib.experiment import lsq_fields


from argparse import ArgumentParser
import importlib.util
import pandas as pd
import h5py

parser = ArgumentParser(
    description = 
        "Fit receptive field model to gradient magnitudes to parameterize" + 
        "receptive fields.")
parser.add_argument('output_path',
    help = 'Path to output RF summary CSV.')
parser.add_argument("rf_grads",
    help = 'Receptive field gradients file, HDF5 Gradients archive.')
parser.add_argument("unit_path",
    help = 'Serialized unit list, must correspond to data in `rf_grads`.')
parser.add_argument("rf_model",
    help = 'Receptive field model file, giving structure whose parameters ' + 
           'will be used to summarize the units\' receptive fields.')
args = parser.parse_args()


# -------------------------------------- Load inputs ----

# Rf model
spec = importlib.util.spec_from_file_location(
    "model", args.rf_model)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
model = model_module
# Unit set in the gradient archive
units = lsq_fields.load_units_from_csv(args.unit_path)
# Gradient archive
grads = h5py.File(args.rf_grads, 'r')


# -------------------------------------- Summarize ----

all_params = []
# Get string unit names organized by latyer
units_serial = vx.VoxelIndex.serialize(units)
for layer in units:
    lstr = '.'.join(str(i) for i in layer)
    grad_key = [k for k in grads.keys() if k.startswith(f'grads_{lstr}')][0]
    params = model.summarize(grads[grad_key][...])
    params = {**params, 'unit': units_serial[layer]}
    all_params.append(params)


# -------------------------------------- Write & Clean up ----

pd.concat([pd.DataFrame(d) for d in all_params]
    ).to_csv(args.output_path, index = False)
grads.close()

