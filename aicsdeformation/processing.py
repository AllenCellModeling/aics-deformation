#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import itertools
import logging
from multiprocessing.dummy import Pool
import numpy as np
import openpiv.process
import openpiv.scaling
import openpiv.validation
import openpiv.filters
from typing import Dict, List, Tuple, Union

from .types import Displacement

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# openpiv reference:
# https://openpiv.readthedocs.io/en/latest/src/tutorial.html


def calculate_displacement(
    frames: Tuple[np.ndarray] = None,
    frame_a: np.ndarray = None,
    frame_b: np.ndarray = None,
    window_size: int = 18,
    overlap: int = 4,
    dt: float = 0.003,
    search_area_size: int = 20,
    sig2noise_method: str = "peak2peak"
) -> Displacement:
    # Expand frames if required
    if frames:
        frame_a, frame_b = (*frames,)

    # Begin calculation
    log.debug("{} Displacement using parameters: [window_size: {}, overlap: {}, dt: {}, search_area_size: {}]".format(
        sig2noise_method, window_size, overlap, dt, search_area_size
    ))
    u, v, sig2noise = openpiv.process.extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=search_area_size,
        sig2noise_method=sig2noise_method
    )
    return Displacement(u, v, sig2noise)


def generate_parameter_searched_displacements(
    frames: Tuple[np.ndarray] = None,
    frame_a: np.ndarray = None,
    frame_b: np.ndarray = None,
    window_size: int = 20,
    window_size_steps: int = 9,
    window_size_step_size: int = 2,
    overlap: int = 4,
    overlap_steps: int = 3,
    overlap_step_size: int = 1,
    dt: float = 0.003,
    dt_steps: int = 5,
    dt_step_size: float = 0.0005,
    search_area_size: int = 18,
    search_area_size_steps: int = 8,
    search_area_size_step_size: int = 2,
    sig2noise_method: str = "peak2peak"
) -> Tuple[Displacement, Dict[Displacement, Dict[str, Union[int, float]]]]:

    # Create all step lists
    window_sizes = set([
        *[window_size - i * window_size_step_size for i in range(window_size_steps)],
        *[window_size + i * window_size_step_size for i in range(window_size_steps)]
    ])

    # Start multithreading
    return window_sizes


def calculate_displacements(frames: List[np.ndarray], n_threads: int = None, **kwargs) -> List[Displacement]:
    # Fence-post: We only want to create displacements for n-frames - 1
    # Additionally enforce frames are numpy types
    frame_pairs = [(frames[i].astype(np.int32), frames[i+1].astype(np.int32)) for i in range(len(frames) - 1)]

    passdown = partial(calculate_displacement, **kwargs)

    # Start multithreading
    with Pool(n_threads) as pool:
        # Map results
        displacements = pool.map(passdown, frame_pairs)

    return displacements
