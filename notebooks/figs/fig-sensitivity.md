
Data files for `fig-sensitivity`:
    - Receptive fields
        - Layer 4: `sensitivity/ell_sn4_n100_b*_ell.csv`
    - Behavior
        - All-layer: `sensitivity/sna_bhv_n300_b*.h5`
        - Layer 1: `sensitivity/sn1_bhv_n300_b*.h5`
        - Layer 2: `sensitivity/sn2_bhv_n300_b*.h5`
        - Layer 3: `sensitivity/sn3_bhv_n300_b*.h5`
        - Layer 4: `sensitivity/sn4_bhv_n300_b*.h5`
    - Effective gain
        - All layers: `sensitivity/lenc_sna_n100_b*.h5.sgain.npz`

Intermediate data files:
    - Receptive fields (gradient form): `sensitivity/`
    - Unit activations: `sensitivity/`

Externally referenced data files:
    - ImageNet composites: `imagenet/imagenet_four224l0.h5`, see `imagenet.md`
    - Linear prediction network heads: `models/logregs_iso224_t100.npz`, see `fig-cornet.md`




### Receptive fields

Focal condition receptive field gradients with the sensitivity shift applied at layer 4 can be computed using `backprop.py`, as in `fig-gauss.md`, by setting the output path to `sensitivity/sn4_rf_n100_b${BETA}.h5` and replacing the `--attn` and `--attn_cfg` arguments with

```bash
    --attn $CODE/proc/att_models/sens_norm.py           `# Attention ` \
    --attn_cfg "layer=[(0,4,0)]:beta=$BETA"              \
```

Due to the computational intensity of the sentitivity-shifted convolution operation, we recommend using the `--batch_size` parameter. On our 16GB GPUs, we were able to use a batch size of 3.

There is no need to compute distributed condition receptive fields again, as they were computed in `fig-gauss.md`. The ellipse-fit summaries can be generated using `summarize_rfs.py` as in `fig-gauss.md` with the new `sn4_rf_n100_b*.h5` files and with the output path set to `sensitivity/ell_sn4_n100_b${BETA}_ell.csv`



### Unit activations and effective gain

Focal unit activations for the sensitivity shift attention applied at all layers can be computed using `encodings.py`, as in `fig-gauss.md`, by setting the output path to `sensitivity/lenc_sna_n100_b${BETA}.h5` and replacing the `--attn` and `--attn_cfg` arguments with

```bash
    --attn $CODE/lib/att_models/sens_norm.py           `# Attention ` \
    --attn_cfg "layer=[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]:beta=$BETA"    \
```

Effective gain may then be computed similarly using `act_dists.py` by passing the new `lenc_sna_n100_b*.h5` files instead of the corresponding `lenc_task_gauss_b*.h5` files.



### Behavior

Behavior on the detection task with the sensitivity shift attention model applied at various combinations of layers can be computed as in `fig-cornet.md`, where behavior was observed under Gaussian gain attention. Here, however, the output path should be replaced with `sensitivity/sn[1|2|3|4|a]_bhv_n300_b*.h5` with the `sn*` chosen according to which layers the attention is applied at, and the use the attention model arguments `--attn` and `--attn_cfg` should be replaced as in the above sections, with the appropriate array of tuples in the `layer` config variable. Below is a bash loop to run all five conditions with moderate batch size on a GPU cluster:

```bash
for COND in 0 1 2 3 4; do
for B in 1.1 2.0 4.0 11.0; do
NAMES=(1 2 3 4 a)
NAME=${NAMES[$COND]}
LAYERS=("[(0,1,0)]" "[(0,2,0)]" "[(0,3,0)]" "[(0,4,0)]" "[(0,1,0),(0,2,0),(0,3,0),(0,4,0)]")
L=${LAYERS[$COND]}
python3 $CODE/script/reg_task.py \
    $DATA/sensitivity/sn${NAME}_bhv_n300_b$B.h5  `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    300                                          `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model $CODE/cornet/cornet_zr.py            `# Observer model` \
    --attn $CODE/lib/att_models/sens_shift.py    `# Attention ` \
    --attn_cfg "layer=$L:beta=$B"  \
    --batch_size 50 \
    --decoders '(0,5,2)'                         `# Decoder layers` \
    --cuda
done; done
```



