# AICS Deformation

[![build status](https://travis-ci.com/AllenCellModeling/aics-deformation.svg?branch=master)](https://travis-ci.com/AllenCellModeling/aics-deformation)
[![codecov](https://codecov.io/gh/AllenCellModeling/aics-deformation/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/aics-deformation)


| **Bead Timeseries Input** | **Deformations Over Cells Output** |
|---------------------------|------------------------------------|
| ![bead layer timeseries](resources/beads.gif =400x) | ![deformations over cells](resource/defs.gif =400x) |

A collection of tools for pre-processing, generating, and post-processing deformation related tasks.

---

## Features
* AICSDeformation object that wraps openPIV to make deformation generation quick and easy.
* Simple interface for deformation parameter searching.
* Pre-process and post-process cleaning, formatting, and visualization tools.

## Overview
1. **[Optional]** Pre-process a CZI time-lapse movie of cells grown on matrix containing beads into (1) a stack of bead
slices and (2) a stack of max-projections. This component also applies Ransac to remove camera jitter from the
time-lapse.

2. **[Core]** Given a list of `numpy.ndarrays` (each `ndarray` being a 2d image), compute the (u, v) deformations at each
(x, y) point.

3. **[Optional]** Post-process the deformation data into a heatmap image and overlay the cell max-projection onto it.
Composite these frames into an mp4 movie.

## Installation
PyPi installation not available at this time, please install using git.

`pip install git+https://github.com/AllenCellModeling/aics-deformation.git`

***Note:*** For the Core components `Numpy` and `Cython` are required to be installed before attempting to install this
package. For the optional components it is required to install `openCV`. Some of these pre-install requirements suggest
source builds. To make life easier it is recommended to use a conda environment and using `conda install -c conda-forge
openpiv numpy opencv`.


### Credits
This package was created with Cookiecutter. [Original repository](https://github.com/audreyr/cookiecutter)


***Free software: Allen Institute Software License***
