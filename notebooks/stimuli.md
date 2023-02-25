TODO after library:
- What is `meta_csv` for `iso_distill`?


Generated data files:
    - Full-size isolated base images `imagenet/imagenet_iso224.h5`
    - Scaled down isolated base images `imagenet/imagenet_iso112.h5`
    - Detection task composite images `imagenet/imagenet_four224l0.h5`


Intermediate data files:
    - Index of ImageNet source images `imagenet/index.csv`


### ImageNet source images

Base images can be downloaded from `http://www.image-net.org/explore`, in `.tar.gz` archives according to their associated WordNet id (for example `n01773797` is "garden spider"). Image files do not need to be in any particular directory structure, but instead are listed in an `index.csv` file with columns: `wnid` (WordNet id), `path`, `n` (an ascending within-category index for the images), `neg_n` (reversed index used for drawing non-overlapping sets of images)

Useful for generating `index.csv` files can be the following commands, which append the CSV columns for a given category.

```bash
WNID=n01773797
ls /ILSVRC_download/$WNID > /tmp/paths.csv
paste -d , <(for _ in $(seq 0 1250); do echo $WNID; done) /tmp/paths.csv <(for _ in $(seq 0 1250); do echo tree; done) <(seq 0 1250) <(seq 1250 0) >> $DATA/imagenet/index.csv
```

Note that ImageNet source images are not square, but in generating the HDF5 archive of base images, a maximal square center crop will be taken. The `index.csv` file should contain the sample category for the human psychophysics task, as the image set for the computational task will be generated later and explicitly exclude that category.


### Combine base images into HDF5 archive

For speed of the dataset generating scripts, we first load all the images into a single HDF5 archive.

```bash
py3 $CODE/script/render_imagenet.py $DATA/imagenet/index.csv $DATA/imagenet/imagenet.h5
```

This script is just an interface for the `generate_dataset` function in `detection_task.py`, which resizes and crops images then stores them in an HDF5 archive according to category.


### Select exemplar images

ImageNet is a great source for its volume of images, but many of the images aren't clear enough members of the category for a human psychophysics experment. For this reason we go through and select a subset of images as "exemplars". To do so, use the following script for each category:
```bash
py3 code/script/imagenet_exemplars.py $DATA/imagenet/imagenet.h5 $DATA/imagenet_exemplars/ "garden spider" 0
```
where the final argument is the index of the first image to prompt with, which can be useful if you need to do this exemplar selection in multiple sessions.


### Human dataset generation

The script to assemble the human trial dataset is used as:
```bash
py3 code/script/build_human_detection_task.py
    <input_directory>       # Directory containing input images
    <image_size>            # Size of one corner of the composite
    <n_positive_focl>       # Num images w/ target in each corner (to be cued)
    <n_positive_dist>       # Num images w/ target in each corner (no cue)
    <n_neg>                 # Num images w/ no target present
    <n_exemplars>           # Number of example images to hold out
    <seed>                  # For replicability           
    <input_image_extension> # Input files named Category_imagenumber.EXT
    <output_directory>      # Where to store output files
    <plot_path>             # Optional path to diagnostic plot
```
for which we assembled the dataset with the command
```bash
py3 code/script/build_human_detection_task.py \
    data/imagenet_exemplars \
    112 \
    5 \
    5 \
    40 \
    5 \
    1 \
    png \
    $DATA/imagenet_human \
    $DATA/imagenet_human/diagnostic.pdf
```
This will create several files in the directory `$DATA/imagenet_human`, which are used by matlab code to run the psychophysics experiment.

### Isolated base images

To create the isolated base images for the computational task used to train the linear network heads, `iso_distill.py` uses the methods defined in `detection_task.py` to generate sets of isolated images and cache them to HDF5 for quick access by later scripts:
```bash
py3 $CODE/script/iso_distill.py $DATA/imagenet/imagenet.h5 /tmp/generated_metdata.csv $DATA/imagenet/imagenet_iso224.h5 224 "tree"
```
After the input and output HDF5 file arguments, pass an integer size for the images, as well as any number of category names to omit from the file (which can be used to stop the model from running on the sample category for the psychophysics experiment).

We also generate scaled down base images to pass to `gen_discrim_composites.py` in `fig-discrimsupp.md`. These are output to `imagenet/imagenet_iso112.h5` using `112` as the size argument.


### Detection task composites

To create the composite images for the neural network observer detection task, `four_distill.py` uses the methods defined in `detection_task.py` to generate sets of composite image grids and cache them to HDF5 for quick access by later scripts:

```bash
py3 $CODE/script/iso_distill.py $DATA/imagenet/imagenet.h5 $DATA/imagenet/imagenet_four224l0.h5 224 "tree"
```




