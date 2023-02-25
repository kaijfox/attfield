from lib.experiment import detection_task as det

from argparse import ArgumentParser, REMAINDER

parser = ArgumentParser(
    description = 
        "Cache isolated base images in an HDF5 archive.")
parser.add_argument("imagenet_archive",
    help = "Path to HDF5 archive genrated by `render_imagenet.py`")
parser.add_argument("meta_csv",
    help = "Output path for image metadata")
parser.add_argument("output_h5",
    help = "Path to output HDF5 archive of isolated base images")
parser.add_argument("image_size", type = int,
    help = "Integer size (square) of images to generate")
parser.add_argument("blacklist", nargs = REMAINDER,
    help = "Names of categories to omit from the generated file")
args = parser.parse_args()

det.cache_iso_task(
    args.imagenet_archive,
    args.output_h5,
    blacklist = args.blacklist,
    metadata_csv = args.meta_csv,
    image_size = args.image_size
)