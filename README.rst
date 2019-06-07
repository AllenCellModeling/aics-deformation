================
aics-deformation
================


.. #image:: https://img.shields.io/pypi/v/aicsdeformation.svg
        :target: https://pypi.python.org/pypi/aicsdeformation

.. image:: https://travis-ci.org/AllenCellModeling/aics-deformation.svg?branch=master
        :target: https://travis-ci.org/AllenCellModeling/aics-deformation

.. #image:: https://readthedocs.org/projects/aicsdeformation/badge/?version=latest
        :target: https://aicsdeformation.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Overview:
---------

Tools used for interacting with AICS deformation projects.

The project has 3 components

1. [Optional] Pre-process a CZI time-lapse movie of cells grown on matrix containing beads into (1) a stack of bead slices and (2) a stack of max-projections. This component also applies Ransac to remove camera jitter from the time-lapse.

2. [Core] Given a list of ndarrays (each ndarray being 2d image), compute the (u, v) deformations at each (x, y) point.

3. [Optional] Post-process the deformation data into a heatmap image and overlay the cell max-projection onto it. Composite these frames into an mp4 movie.


Pre-Install Requirements:
_________________________

* For the Core components Numpy and Cython are required to be installed before attempting to install this package.

* For the optional components it is required to install openPIV and openCV.

Some of these pre-install requirements suggest source builds. To make life easier I would recommend using a conda
environment and using

``conda install -c conda-forge openpiv numpy opencv``

Installation:
-------------

``pip install git+https://github.com/AllenCellModeling/aics-deformation.git``


Reference Info:
---------------

* Free software: Allen Institute Software License

* Documentation: https://aicsdeformation.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
