
Post-library TODO:
- Reference correct script for 1x1 scores
- Apply name changes

Data files for `fig-gauss`:
    - Receptive fields
        - `gauss/summ_base_ell.csv`
        - `gauss/summ_cts_gauss_b*_ell.csv`
    - Effective gain
        - `gauss/lenc_task_gauss_b*.h5.sgain.npz` 
    - Receptive fields (raw gradient form)
        - `gauss/rfs_base.h5`
        - `gauss/rfs_cts_gauss_beta_11.0.h5`
    - Effective gain after nonlinear normalization
        - `gauss/enc_l1_base.h5.norm.sd.npz`
        - `gauss/enc_l1_gauss_b*.h5.norm.sgain.npz`
    - Layer 4 feature map 1x1 readout AUCs
        - `gauss/1x1_scores.npz`

Intermediate data files:
    - Subset of neural units: `models/cZR_300units_rad.csv`
    - Unit activations
        - `gauss/lenc_task_base.h5`
        - `gauss/lenc_task_gauss_b*.h5`
    - Post-nonlinearity layer 1 unit activations
        - `gauss/enc_l1_base.h5`
        - `gauss/enc_l1_gauss_b*.h5`

External referenced data files
    - ImageNet composites: `imagenet/imagenet_four224l0.h5`, see `imagenet.md`
    - Linear classification layers: `models/logregs_iso224_t100.npz`, see `fig-cornet.md`
    - Full feature-map linear prediction layers: `apool/ign_iso224_coefs.npz`, see `fig-avgsupp.md`
    - Distributed condition activations: `apool/enc_task_imgnet_base.h5`, see `fig-avgsupp.md`
    - Activations under 4x Gaussian gain: `apool/enc_task_imgnet_gauss_b4.0.h5`, see `fig-avgsupp.md`


### Subset of neural units

It would be computationally prohibitive to study the receptive fields and effective gain of all units in CORnet, so select a subset of 300 units per layer.

```bash
python3 $CODE/script/gen_radial_units.py \
    $DATA/models/cZR_300units_rad.csv            `# Output Path` \
    300                                          `# Units per layer` \
    "(0,1,0)" "(0,2,0)" "(0,3,0)" "(0,4,0)"      `# Layers to sample` \
    --loc 0.25 0.25                              `# Attention locus` \
    --size 224 --channels 3                      `# Expected input dims` \
    --model_file $CODE/cornet/cornet_zr.py `         # Model`
```

### Receptive fields

Calculate gradients from selected units back to the input tensor with no attention model. All gradient calculations should be done on GPUs, but the ellipse fits afterwords may be done on CPU.

```bash
python3 $CODE/script/backprop.py \
    $DATA/gauss/rfs_base.h5                             `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
    100                                                 `# Imgs per category` \
    $DATA/models/cZR_300units_rad.csv                   `# Unit set` \
    '(0,0)'                                             `# Gradients w.r.t.` \
    --model $CODE/cornet/cornet_zr.py                   `# Model` \
    --abs                                               `# Absolute grads ` \
    --decoders '(0,5,2)'                                `# Decoder layers` \
    --batch_size 300                                    `# Limit memory` \
    --verbose                                           `# Debug` \
    --cuda
```

Calculate gradients from selected units back to the input tensor at four strengths of Gaussian gain attention:

```bash
for BETA in 1.1 2.0 4.0 11.0; do
    python3 $CODE/script/backprop.py \
        $DATA/gauss/rfs_cts_gauss_beta_$BETA.h5             `# Output Path` \
        $DATA/imagenet/imagenet_four224l0.h5                `# Image Set` \
        100                                                 `# Imgs per category` \
        $DATA/models/cZR_300units_rad.csv                   `# Unit set` \
        '(0,0)'                                             `# Gradients w.r.t.` \
        --attn $CODE/lib/att_models/cts_gauss_gain.py       `# Model type` \
        --attn_cfg $CODE/lib/att_models/retina_b$BETA.json  `# Model params` \
        --model $CODE/cornet/cornet_zr.py                   `# Model` \
        --abs                                               `# Absolute grads ` \
        --decoders '(0,5,2)'                                `# Decoder layers` \
        --batch_size 300                                    `# Limit memory` \
        --verbose                                           `# Debug` \
        --cuda
done
```

Summarize gradient data into ellipses by fitting a Gaussian distribution:

```bash
python3 $CODE/script/summarize_rfs.py \
    $DATA/gauss/summ_base_ell.csv              `# Output Path` \
    $DATA/gauss/rfs_base.h5                    `# RF Gradients` \
    $DATA/models/cZR_300units_rad.csv          `# Unit set` \
    $CODE/lib/rf_models/gaussian.py            `# RF Model`

for BETA in 1.1 2.0 4.0 11.0; do
python3 $CODE/script/summarize_rfs.py \
    $DATA/gauss/summ_cts_gauss_b${BETA}_ell.csv       `# Output Path` \
    $DATA/gauss/rfs_cts_gauss_beta_${BETA}.h5         `# RF Gradients` \
    $DATA/models/cZR_300units_rad.csv                 `# Unit set` \
    $CODE/lib/rf_models/gaussian.py                   `# RF Model`
done
```

### Unit activations

Record activations of units at each layer without any attention model, so that we can compare these to activations when attention is applied and measure effective gain.

```bash
IMG=$DATA/imagenet/imagenet_four224l0.h5

python3 $CODE/script/encodings.py \
    $DATA/gauss/lenc_task_base.h5                   `# Output Path` \
    $CODE/lib/image_gen/det_task.py                 `# Image Set` \
    $CODE/cornet/cornet_zr.py                       `# Model` \
    "(0,1,0)" "(0,2,0)" "(0,3,0)" "(0,4,0)"         `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --cuda
```

Record activations with Gaussian gain attention at the standard four strengths:

```bash
for BETA in 1.1 2.0 4.0 11.0; do
echo "BETA:" $BETA
python3 $CODE/script/encodings.py \
    $DATA/gauss/lenc_task_gauss_b${BETA}.h5         `# Output Path` \
    $CODE/lib/image_gen/det_task.py                 `# Image Set` \
    $CODE/cornet/cornet_zr.py                       `# Model` \
    "(0,1,0)" "(0,2,0)" "(0,3,0)" "(0,4,0)"         `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --attn $CODE/lib/att_models/cts_gauss_gain.py   `# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=$BETA"            \
    --cuda
done
```



### Effective gain

Compare unit activations in the distributed conditions to those under each Gaussian gain attention strength. Note that the `act_dists.py` script generates a diagnostic plot showing gain under each condition at each layer, but also generates data files with the suffix `.sgain.npz` (short for standard deviation gain) containing processed effective gain measurements.

```bash
python3 $CODE/script/plots/act_dists.py \
    $PLOTS/figures/fig2/line_gain_gauss_ylim.pdf      `# Output Path` \
    $DATA/gauss/lenc_task_base.h5                     `# Base Acts` \
    $DATA/gauss/lenc_task_gauss_b1.1.h5               `# Cued RFs` \
    $DATA/gauss/lenc_task_gauss_b2.0.h5  \
    $DATA/gauss/lenc_task_gauss_b4.0.h5  \
    $DATA/gauss/lenc_task_gauss_b11.0.h5  \
    --loc .25 .25 --rad .25 --degrees_in_img 1        `# Locus of attention` \
    --disp "1.1" "2.0" "4.0" "11.0"                   `# Plotting arguments` \
    --pal_f $DATA/cfg/pal_beta.csv \
    --pal_l $DATA/cfg/pal_layer.csv \
    --raw_ylim 0.2 1e6 --sd_ylim .2 15 \
    --figsize 5 5 \
    --n_img 3                                         `# Long op's on subset of imgs`
    --no_read                                         `# Do not use cached .sgain files`
```


### Effective gain after nonlinear normalization

Record activations of layer 1 units after nonlinearities without an attention model present, and with a range of low-amplitude Guassian gain attention models:

```bash
IMG=$DATA/imagenet/imagenet_four224l0.h5

# distributed
python3 $CODE/script/encodings.py \
    $DATA/gauss/enc_l1_base.h5        `# Output Path` \
    $CODE/lib/image_gen/det_task.py   `# Image Set` \
    $CODE/cornet/cornet_zr.py         `# Observer model` \
    "(0,1,3)"                         `# Pull layer` \
    --gen_cfg "img=$IMG:n=50"         `# Image config`

# focal
for BETA in 1.00 1.01 1.02 1.03 1.04 1.05 1.06 1.07 1.08 1.09 1.10; do
echo "BETA:" $BETA
python3 $CODE/script/encodings.py \
    $DATA/gauss/enc_l1_gauss_b${BETA}.h5            `# Output Path` \
    $CODE/lib/image_gen/det_task.py                 `# Image Set` \
    $CODE/cornet/cornet_zr.py                       `# Observer model` \
    "(0,1,3)"                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=50"                       `# Image config` \
    --attn $CODE/lib/att_models/cts_gauss_gain.py   `# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=$BETA"
done
```

Compute a precursor to effective gains that can be used to measure effective gain under normalization quickly during plotting and store them in `.norm.sgain.npz` files adjacent to the focal activation files.

```bash
python3 $CODE/script/norm_heatmap.py \
    $PLOTS/figures/fig2/norm_shallow.pdf  `# Output` \
    $DATA/gauss/enc_l1_base.h5            `# Distributed Activations` \
    $DATA/gauss/enc_l1_gauss_b1.00.h5     `# Focal activations` \
    $DATA/gauss/enc_l1_gauss_b1.01.h5  \
    $DATA/gauss/enc_l1_gauss_b1.02.h5  \
    $DATA/gauss/enc_l1_gauss_b1.03.h5  \
    $DATA/gauss/enc_l1_gauss_b1.04.h5  \
    $DATA/gauss/enc_l1_gauss_b1.05.h5  \
    $DATA/gauss/enc_l1_gauss_b1.06.h5  \
    $DATA/gauss/enc_l1_gauss_b1.07.h5  \
    $DATA/gauss/enc_l1_gauss_b1.08.h5  \
    $DATA/gauss/enc_l1_gauss_b1.09.h5  \
    $DATA/gauss/enc_l1_gauss_b1.10.h5  \
    --exponent $(seq 1 0.2 4) \
    --loc .25 .25 --rad .25 --degrees_in_img 1 \
    --n_img 30 --n_feat 30 \
    --layers "0.1.3" \
    --loc_field 0.02 0.5 \
    --disp 1.00 1.01 1.02 1.03 1.04 1.05 1.06 1.07 1.08 1.09 1.10 \
    --figsize 5 5 --no_line
```


### Layer 4 feature alignment with decision axes

To compute the alignment of features from each spatial location in layer 4 via their AUC's as a 1x1 readout, use `location_readouts.py`

```bash
python3 $CODE/script/location_readouts.py \
    $DATA/gauss/1x1_scores.npz                  `# Output path` \
    $DATA/apool/apool/ign_iso224_coefs.npz      `# Full map classifiers` \
    $DATA/apool/enc_task_imgnet_base.h5         `# Dist. activations` \
    $DATA/apool/enc_task_imgnet_gauss_b4.0.h5   `# Focal activations`
```