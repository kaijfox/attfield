
Data files for `fig-avgsupp`:
    - Masked readout behavior: `apool/4x4_neccsuff_scores.npz`
    - Full feature-map linear prediction layers: `apool/ign_iso224_coefs.npz`
    - Individual feature map location behavior: `apool/7x7_neccsuff_scores.npz`

Intermediate data files:
    - Isolated image activations: `apool/apool/enc_ign_iso224.h5`
    - Distributed condition activations: `apool/enc_task_imgnet_base.h5`
    - Activations under 4x Gaussian gain: `apool/enc_task_imgnet_gauss_b4.0.h5`

External referenced data files
- ImageNet composites: `imagenet/imagenet_four224l0.h5`, see `imagenet.md`


### Distributed condition activations

Unit activations in the distributed condition can be calculated as in `fig-gauss.md`, using the `encodings.py` script, but using the following command are recorded only from *after* the nonlinearity in the fourth layer, precisely as they would be passed to the average pooling and classifier layers. Note that this command applies the same operation as is used in `fig-reconstruct.md` on a slightly larger image set.

```bash
IMG=$DATA/imagenet/imagenet_four224l0.h5
N_IMG_PER_CAT=300
py3 $CODE/script/encodings.py \
    $DATA/apool/enc_task_imgnet_base.h5             `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    $CODE/cornet/cornet_zr.py                       `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=$N_IMG_PER_CAT"           `# Image config`
```

### Focal condition activations

Corresponding unit activations to those above, but from a focal condition with 4X Gaussian gain applied at the first layer, may be computed by adding attention parameters `--attn` and `--attn_cfg` to the `encodings.py` call, as in `fig-gauss.md` and `fig-reconstruct.md`:

```bash
IMG=$DATA/imagenet/imagenet_four224l0.h5
N_IMG_PER_CAT=300
BETA=4.0
py3 $CODE/script/encodings.py \
    $DATA/apool/enc_task_imgnet_gauss_b$BETA.h5     `# Output Path` \
    $CODE/proc/image_gen/det_task.py                `# Image Set` \
    $CODE/cornet/cornet_zr.py                       `# Model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=$N_IMG_PER_CAT"           `# Image config` \
    --attn $CODE/proc/att_models/cts_gauss_gain.py  `# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=$BETA"
```


### Isolated image activations and full feature map prediction layers

To compute statistics on the effect of average pooling in readout, and to provide classifiers (that *do* act on average-pooled activations) to later scripts in appropriate format, use `encodings.py` to generate a large set of activations on isolated images:

```bash
N_IMG_PER_CAT=1800
py3 $CODE/script/encodings.py \
    $DATA/apool/enc_ign_iso224.h5                `# Output Path` \
    $CODE/proc/image_gen/h5_images.py            `# Image Set` \
    code/cornet/cornet/cornet_zr.py              `# Model` \
    '(0,4,3)'  --batch_size 200                  `# Pull layer` \
    --gen_cfg "img=$IMG:n=$N_IMG_PER_CAT"        `# Image config`
```

Then train pooled and full-map logistic regressions with `train_fullmap_logregs.py`:

```bash
py3 code/script/train_fullmap_logregs.py \
    $DATA/apool/ign_iso224_coefs.npz        `# Output path` \
    $DATA/apool/enc_ign_iso224.h5           `# Activations` \
    '0.4.3'                                 `# Layer` \
    1600                                    `# Num training imgs`

```


### Masked readout behavior

Scores in the distributed, focal, propagated gain map multiplied, and divided conditions - as in `fig-reconstruct.md` can be computed using `masked_readout_task.py` as follows:

```bash
py3 code/script/masked_readout_task.py \
    $DATA/apool/4x4_neccsuff_scores.npz         `# Output path` \
    $DATA/apool/apool/ign_iso224_coefs.npz      `# Full map classifiers` \
    $DATA/apool/enc_task_imgnet_base.h5         `# Dist. activations` \
    $DATA/apool/enc_task_imgnet_gauss_b4.0.h5   `# Focal activations` \
    4                                           `# Mask size`
```