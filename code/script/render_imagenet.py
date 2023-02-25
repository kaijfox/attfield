from lib.experiment import detection_task as det

from argparse import ArgumentParser, REMAINDER

parser = ArgumentParser(
    description = 
        "Cache isolated base images in an HDF5 archive.")
parser.add_argument("index_file",
    help = "Path to CSV index of imagenet sources - see `stimuli.md`.")
parser.add_argument("output_h5",
    help = "Path to output HDF5 archive.`")
parser.add_argument("blacklist", nargs = REMAINDER,
    help = "Names of categories to omit from the generated file")
args = parser.parse_args()


if __name__ == '__main__':
	det.generate_dataset(
		args.index_file,
		args.output_h5,
		blacklist = args.blacklist)
