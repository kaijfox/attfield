

## Behavioral benefits of spatial attention explained by multiplicative gain, not receptive field shifts, in a neural network model.


This repository contains the implementation of attention models in: Kai Fox, Dan Birman, and Justin Gardner. “Behavioral benefits of spatial attention explained by multiplicative gain, not receptive field shifts, in a neural network model.” bioRxiv. 2022. [[preprint](https://www.biorxiv.org/content/10.1101/2022.03.04.483026v1)].

The main contents are

- `code/script/figs/` – Scripts to quickly generate each figure of the manuscript using pre-generated data. The order of the figures as presented in the manuscript is `fig-task`, `fig-cornet`, `fig-gauss`, `fig-shift`, `fig-sensitivity`, `fig-shrink`\*, `fig-flat`, `fig-reconstruct` (w/ supplements `fig-avgsupp`\* and `fig-discrimsupp`\*). Those listed with \* appear only in the revised manuscript.
- `code/script/` – Scripts to generate data for figures, intended to be run in a high-performance-computing (HPC) environment.
- `code/lib` – Classes and methods to support the mechanistic models of attention studied in the figures.
- `notebooks/` – Markdown files for each figure describing the use of the HPC scripts to generate data referenced in the corresponding `code/script/figs` file.


#### Setup

This repo is built mainly on `torch`, `numpy`, `matplotlib`, and `h5py`, but a full list of libraries and versions for setup using your virtual environment manager of choice may be found in `requirements.txt`

Much of the code is designed to be run across a combination of local and high-performance-computing environments, so all paths are specified relative to `CODE` and `DATA` environment variables (accessed in python via `lib.paths`) that should be defined in a `bash_rc` or other environment startup script.