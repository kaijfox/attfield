

Data files for `fig-discrimsupp`:
    - Category discrimination classifiers: `discrim/regs_ign224_pair.npz`
    - Distributed condition activations: `discrim/enc_ign_cls_tifc.h5`
    - Focal condition activations: `discrim/enc_ign_cls_tifc_b4.0.h5`

Intermediate data files:
    - Category discrimination task composites: `imagenet/imagenet_cls_tifc.h5`

External referenced data files:
    - Isolated image encodings: `apool/enc_ign_iso224.h5`, see `fig-avgsupp.md`.
    - Scaled-down isolated images: `imagenet/imagenet_iso112.h5`, see `stimuli.md`.



### Category discrimination classifiers

A new set of linear prediction network heads are needed for the category discrimination task, which can be trained with `train_pair_logregs.py`:

```bash
py3 code/sript/train_pair_logregs.py \
    $DATA/discrim/regs_ign224_pair.npz      `# Output path` \
    $DATA/apool/enc_ign_iso224.h5           `# Isolated image encodings` \
    '0.4.3'                                 `# Layer` \
    400 200                                 `# Num train and val images`
```

### Category discrimination task stimuli

Similarly to what was done in `stimuli.md` for the main detection task studied, we must turn isolated images into composites for the category discrimination task. This can be done using `gen_discrim_composites.py`

```bash
py3 code/script/gen_discrim_composites.py \
    $DATA/imagenet/imagenet_cls_tifc.h5     `# Output path` \
    $DATA/imagenet/imagenet_iso112.h5       `# Scaled-down isolated images` \
    450                                     `# Num composites per category` \
    400                                     `# Starting image index`
```


### Focal and distributed condition activations

Stimuli for the category discrimination task must be converted to resulting layer-4 encodings in distributed and focal (Gaussian 4x gain) conditions to calculate behavioral results in `fig-discrimsupp.py`, using `encodings.py` as in `fig-avgsupp.md` and `fig-reconstruct.md`:

```bash
IMG=$DATA/imagenet/imagenet_cls_tifc.h5
N_IMG_PER_CAT=900
$py3 $CODE/script/encodings.py \
    $DATA/discrim/enc_ign_cls_tifc.h5            `# Output Path` \
    $CODE/lib/image_gen/h5_images.py             `# Image Set` \
    $CODE/cornet/cornet_zr.py                    `# Model` \
    '(0,4,3)'  --batch_size 200                  `# Pull layer` \
    --gen_cfg "img=$IMG:n=$N_IMG_PER_CAT"        `# Image config`

BETA=4.0
py3 $CODE/script/encodings.py \
    $DATA/discrim/enc_ign_cls_tifc_b$BETA.h5    `# Output Path` \
    $CODE/lib/image_gen/h5_images.py            `# Image Set` \
    $CODE/cornet/cornet_zr.py                   `# Model` \
    '(0,4,3)'  --batch_size 200                 `# Pull layer` \
    --gen_cfg "img=$IMG:n=$N_IMG_PER_CAT"       `# Image config` \
    --attn $CODE/lib/att_models/cts_gauss_gain.py`# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=$BETA"
```