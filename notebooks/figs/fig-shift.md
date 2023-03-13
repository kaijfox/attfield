
Data files for `fig-shift`:
    - Receptive fields
        - `shift/summ_mim_gauss_b*_ell.csv`
    - Behavior
        - `shift/bhv_mim_gauss_beta_*.h5`
    - Effective gain
        - `shift/lenc_mg_n100_b*.h5.sgain.npz`

Intermediate data files:
    - Shift maps: `models/field_gauss_b*.h5`
    - Receptive fields (gradient form): `shift/rfs_mim_gauss_b*.h5`
    - Unit activations: `shift/lenc_mg_n100_b*.h5`




### Shift maps

To determine desired receptive field shifts and therefore how to rewire units, estimate interpolated maps of receptive field shifts (also called shift fields in the code) for each strength of Gaussian gain attention.

```bash
for B in 1.1 2.0 4.0 11.0
do
python3 code/script/gen_unit_field.py \
    $DATA/models/field_gauss_b${B}.h5         `# Output Path` \
    $DATA/guass/summ_cts_gauss_b${B}_ell.csv  `# Cued RFs` \
    $DATA/gauss/summ_base_ell.csv             `# Base RFs` \
    "(0, 4, 0)"                               `# Layer to mimic` \
    224                                       `# Input space size`
done
```

### Receptive fields

Focal condition receptive field gradients can be computed using `backprop.py`, as in `fig-gauss.md`, by setting the output path to `shift/rfs_mim_gauss_b$BETA.h5` and replacing the `--attn` and `--attn_cfg` arguments with

```bash
    --attn $CODE/lib/att_models/field_shift.py        `# Model type` \
    --attn_cfg "layer=(0,1,0):beta=1.0:field_file=$DATA/models/field_gauss_b${BETA}.h5"  `# Model params` \
```

Due to the computational intensity of the rewired convolution operation, we recommend using the `--batch_size` parameter. On our GPU cluster, we were able to use a batch size of `9.01` -- floating point batch sizes in `backprop.py` indicate number of images per GB of GPU memory on the device.

There is no need to compute distributed condition receptive fields again, as they were computed in `fig-gauss.md`. The ellipse-fit summaries can be generated using `summarize_rfs.py` as in `fig-gauss.md` with the new `rfs_mim_gauss_b*.h5` files and with the output path set to `shift/summ_mim_gauss_b${BETA}_ell.csv`



### Unit activations and effective gain

Unit activations in the focal condition can be computed using `encodings.py`, as in `fig-gauss.md`, by setting the output path to `shift/lenc_mg_n100_b${BETA}.h5` and replacing the `--attn` and `--attn_cfg` arguments as in the previous section.

Effective gain may then be computed similarly using `act_dists.py` by passing the new `lenc_mg_n100_b*.h5` files instead of the corresponding `lenc_task_gauss_b*.h5` files.


### Behavior

Behavior on the detection task under the shift-only attention model can be computed as in the Gaussian gain attention condition from `fig-cornet.md`. Instead, set the output path to `shift/bhv_mim_gauss_beta_${B}.h5` and use the attention model arguments `--attn` and `--attn_cfg` from the above section on computing receptive fields under the shift-only attention model (replacing the `BETA` shell variable with `B` for compatibility with the for loop in `fig-cornet.md`).

