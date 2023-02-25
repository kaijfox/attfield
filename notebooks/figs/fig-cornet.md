
Post-library TODO:



Data files for `fig-cornet`:
- Distributed condition behavior
    - `cornet/bhv_dist.h5`
- Gaussian gain attention behavior
    - `cornet/bhv_gauss_b*.h5`

Intermediate data files:
- Linear prediction network heads: `models/logregs_iso224_t100.npz`

External referenced data files
- ImageNet composites: `imagenet/imagenet_four224l0.h5`, see `imagenet.md`


### Linear prediction heads for detection task

To measure behavioral results on the detection task, train linear prediction network heads for the observer model:

```bash
py3 $CODE/script/train_logregs.py \
    $DATA/models/logregs_iso224_t100.npz            `# Output Path` \
    $DATA/imagenet/imagenet_iso224.h5               `# Image Set` \
    100                                             `# Num images` \
    "(0,4,2)"                                       `# Decoder layers` \
    --model $CODE/cornet/cornet_zr.py
```


### Distributed condition behavior

As a reference point for all attention models, record behavior of the neural network observer on the detection task without any attention modification.

```bash
py3 $CODE/script/reg_task.py \
    $DATA/cornet/bhv_dist.h5                     `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    300                                          `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model $CODE/cornet/cornet_zr.py            `# Observer model` \
    --decoders '(0,5,2)'                         `# Decoder layers`
```

### Gaussian gain attention behavior

To test whether Gaussian gain attention can enhance performance, run the neural network observer with Gaussian gain attention through the detection task.

```bash
for B in 1.1 2.0 4.0 11.0
do 
py3 $CODE/script/reg_task.py \
    $DATA/cornet/bhv_gauss_b$B.h5                `# Output Path` \
    $DATA/imagenet/imagenet_four224l0.h5         `# Image Set` \
    300                                          `# Imgs per category` \
    $DATA/models/logregs_iso224_t100.npz         `# Regressions` \
    --model $CODE/cornet/cornet_zr.py            `# Observer model` \
    --attn $CODE/att_models/cts_gauss_gain.py    `# Attention ` \
    --attn_cfg "layer=(0,1,0):beta=$B"             \
    --decoders '(0,5,2)'                         `# Decoder layers` 
done
```
