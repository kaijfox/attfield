
Data files for `fig-reconstruct`:
    - Distributed condition activations: `reconst/fnenc_task_base.h5`
    - Activations under 4x Gaussian gain: `reconst/enc_task_gauss_b4.0.h5`

Externally referenced data files:
    - ImageNet composites: `imagenet/imagenet_four224l0.h5`, see `imagenet.md`
    - Linear prediction network heads: `models/logregs_iso224_t100.npz`, see `fig-cornet.md`

### Distributed condition activations

Unit activations in the distributed condition can be calculated as in `fig-gauss.md`, using the `encodings.py` script, but using the following command are recorded only from *after* the nonlinearity in the fourth layer, precisely as they would be passed to the average pooling and classifier layers:

```bash
IMG=$DATA/imagenet/imagenet_four224l0.h5
python3 $CODE/script/encodings.py \
    $DATA/reconst/fnenc_task_base.h5                `# Output Path` \
    $CODE/lib/image_gen/det_task.py                 `# Image Set` \
    $CODE/cornet/cornet_zr.py                       `# Observer model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --regs $DATA/models/logregs_iso224_t100.npz     `# Regressions`
```

### Focal condition activations

Corresponding unit activations to those above, but from a focal condition with 4X Gaussian gain applied at the first layer, may be computed by adding attention parameters `--attn` and `--attn_cfg` to the `encodings.py` call, as in `fig-gauss.md`:

```bash
IMG=$DATA/imagenet/imagenet_four224l0.h
python3 $CODE/script/encodings.py \
    $DATA/reconst/enc_task_gauss_b2.0.h5            `# Output Path` \
    $CODE/lib/image_gen/det_task.py                 `# Image Set` \
    $CODE/cornet/cornet_zr.py                       `# Observer model` \
    '(0,4,3)'                                       `# Pull layer` \
    --gen_cfg "img=$IMG:n=100"                      `# Image config` \
    --regs $DATA/models/logregs_iso224_t100.npz     `# Regressions`
    --attn $CODE/lib/att_models/cts_gauss_gain.py   `# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=4.0"
```