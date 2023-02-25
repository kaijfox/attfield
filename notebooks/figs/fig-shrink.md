
Data files for `fig-shrink`:
    - Receptive fields
        - `shrink/rfs_shrink_b*_ell.csv`
    - Behavior
        - `shrink/bhv_shrink_beta_*.h5`
    - Effective gain
        - `shrink/lenc_ms_n100_b*.h5.sgain.npz`

Intermediate data files:
    - Receptive fields (gradient form): `shrink/rfs_shrink_b*.h5`
    - Unit activations: `shrink/lenc_ms_n100_b*.h5`



### Receptive fields

Focal condition receptive field gradients can be computed using `backprop.py`, as in `fig-gauss.md`, by setting the output path to `shrink/rfs_shrink_b$BETA.h5` and replacing the `--attn` and `--attn_cfg` arguments with

```bash
    --attn $CODE/lib/att_models/manual_shrink.py       `# Model type` \
    --attn_cfg "layer=(0,4,0):beta=$BETA"               `# Model params`\
```

There is no need to compute distributed condition receptive fields again, as they were computed in `fig-gauss.md`. The ellipse-fit summaries can be generated using `summarize_rfs.py` as in `fig-gauss.md` with the new `rfs_shrink_b*.h5` files and with the output path set to `shrink/rfs_shrink_b${BETA}_ell.csv`



### Unit activations and effective gain

Unit activations in the focal condition can be computed using `encodings.py`, as in `fig-gauss.md`, by setting the output path to `shrink/lenc_ms_n100_b${BETA}.h5` and replacing the `--attn` and `--attn_cfg` arguments as in the previous section.

Effective gain may then be computed similarly using `act_dists.py` by passing the new `lenc_ms_n100_b*.h5` files instead of the corresponding `lenc_task_gauss_b*.h5` files.


### Behavior

Behavior on the detection task under the shrink-only attention model can be computed as in the Gaussian gain attention condition from `fig-cornet.md`. Instead, set the output path to `shrink/bhv_shrink_beta_${B}.h5` and use the attention model arguments `--attn` and `--attn_cfg` from the above section on computing receptive fields under the shrink-only attention model (replacing the `BETA` shell variable with `B` for compatibility with the for loop in `fig-cornet.md`).

