#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from multiprocessing.dummy import Pool
import numpy as np
# import openpiv.process
# import openpiv.scaling
# import openpiv.validation
# import openpiv.filters
from typing import List, Tuple

from .types import Displacement

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# openpiv reference:
# https://openpiv.readthedocs.io/en/latest/src/tutorial.html


def calculate_displacement(
    frameA: np.ndarray = None,
    frameB: np.ndarray = None,
    frames: Tuple[np.ndarray] = None,
    window_size: int = 20,

) -> Displacement:
    # Expand tuple if neccessary
    # if frames:
    #     frameA, frameB = (*frames,)

    # Begin calculation
    x = None
    y = None
    u = None
    v = None
    mask = None
    sig2noise = None
    # u, v, sig2noise = openpiv.process.extended_search_area_piv(
    #     frameA,
    #     frameB,
    #     window_size=window_size,
    #     overlap=self.overlap_size,
    #     dt=self.dt,
    #     search_area_size=self.search_area,
    #     sig2noise_method=self.s2n_method
    # )
    # x, y = openpiv.process.get_coordinates(image_size=frame_a.shape, window_size=self.window_size,
    #                                        overlap=self.overlap_size)
    # u, v, mask = openpiv.validation.sig2noise_val(u, v, sig2noise, threshold=self.s2n_thresh)
    # u, v = openpiv.filters.replace_outliers(u, v, method=self.outlier_method, max_iter=10, kernel_size=2)
    # x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor=96.52)
    return Displacement(x, y, u, v, mask, sig2noise)


def calculate_displacements(frames: List[np.ndarray], n_threads: int = None) -> List[Displacement]:
    # Fence-post: We only want to create displacements for n-frames - 1
    frame_pairs = [(frames[i], frames[i+1]) for i in range(len(frames) - 1)]

    # Start multithreading
    with Pool(n_threads) as pool:
        # Map results
        displacements = pool.map(calculate_displacement, frame_pairs)

    return displacements
