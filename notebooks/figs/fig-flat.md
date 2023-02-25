
Data files for `fig-flat`:
    - Receptive fields
        - `flat/summ_stitch_b*_ell.csv`
    - Behavior
        - `flat/bhv_stitch_beta_*.h5`
    - Effective gain
        - `flat/enc_stitch_b*.h5.sgain.npz`

Intermediate data files:
    - Receptive fields (gradient form): `flat/rfs_mim_gauss_b*.h5`
    - Unit activations: `flat/lenc_mg_n100_b*.h5`



### Receptive fields

Focal condition receptive field gradients can be computed using `backprop.py`, as in `fig-gauss.md`, by setting the output path to `flat/rfs_mim_gauss_b$BETA.h5` and replacing the `--attn` and `--attn_cfg` arguments with

```bash
    --attn $CODE/lib/att_models/stitched.py            `# Attention model` \
    --attn_cfg "$SPL:$MRG:beta=1+($BETA-1)*0.35"         \
```

There is no need to compute distributed condition receptive fields again, as they were computed in `fig-gauss.md`. The ellipse-fit summaries can be generated using `summarize_rfs.py` as in `fig-gauss.md` with the new `rfs_stitch_beta_*.h5` files and with the output set to path to `flat/summ_stitch_b${BETA}_ell.csv`



### Unit activations and effective gain

Unit activations in the focal condition can be computed using `encodings.py`, as in `fig-gauss.md`, by setting the output path to `flat/enc_stitch_b${BETA}.h5` and replacing the `--attn` and `--attn_cfg` arguments as in the previous section.

Effective gain may then be computed similarly using `act_dists.py` by passing the new `enc_stitch_b*.h5` files instead of the corresponding `lenc_task_gauss_b*.h5` files.


### Behavior

Behavior on the detection task under the flat-only attention model can be computed as in the Gaussian gain attention condition from `fig-cornet.md`. Instead, set the output path to `flat/bhv_stitch_beta_${B}.h5` and use the attention model arguments `--attn` and `--attn_cfg` from the above section on computing receptive fields under the flat-only attention model (replacing the `BETA` shell variable with `B` for compatibility with the for loop in `fig-cornet.md`).

