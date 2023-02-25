from matplotlib import pyplot as plt
import skimage.io
import h5py
import sys
import os

from argparse import ArgumentParser

parser = ArgumentParser(
    description = 
        "Run a GUI for selecting category exemplar images from the full " +
        "ImageNet set - see `stimuli.md`.")
parser.add_argument("imagenet_archive",
    help = "Path to HDF5 archive genrated by `render_imagenet.py`")
parser.add_argument("out_dir",
    help = "Output directory for chosen exemplar images.")
parser.add_argument("category",
    help = "Category to select exemplars for.")
parser.add_argument("start_ix", type = int, default = 0,
    help = "Starting index in the category for the GUI.")
args = parser.parse_args()



imagenet_h5 = h5py.File(args.imagenet_archive, 'r')
c = args.category



imgs = imagenet_h5[c][...]
exemplars = set()

# Iterate over images and ask whether they are exemplars
for i in range(args.start_ix, len(imgs)):
	plt.imshow(imgs[i])
	plt.title(f'{c} : {i}')
	plt.axis('off')
	plt.draw()
	plt.pause(0.01)

	action = ""
	while action not in ['e', 'n', 's']:
		action = input(
			f'[n={len(exemplars)}] ' + 
			'Action? ([e]xemplar, [n]on-Exemplar, [s]top-Category) ')

	plt.close()
	if action == 'e':
		exemplars.add(i)
	elif action == 's':
		break

# Arrange and output exemplars
if not os.path.isdir(args.out_dir):
	os.mkdir(args.out_dir)
for i_e, e in enumerate(exemplars):
	skimage.io.imsave(f'{args.out_dir}/{c}_{e}.png', imgs[e].astype('uint8'))

imagenet_h5.close()

