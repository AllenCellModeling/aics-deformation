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
from typing import Dict, List, NamedTuple, Set, Tuple, Union
from tqdm import tqdm

from .types import Displacement

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# openpiv reference:
# https://openpiv.readthedocs.io/en/latest/src/tutorial.html


class DisplacementFromParameters(NamedTuple):
    displacement: Displacement
    parameters: Dict[str, Union[int, float]]


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

    # Handle numpy types
    frame_a = frame_a.astype(np.int32)
    frame_b = frame_b.astype(np.int32)

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


def _generate_window(center: Union[int, float], steps: int, step_size: Union[int, float]) -> Set[Union[int, float]]:
    window = set([
        *[center - i * step_size for i in range(steps + 1)],
        *[center + i * step_size for i in range(steps + 1)]
    ])

    return window


def grid_search_displacements(
    frames: Tuple[np.ndarray] = None,
    frame_a: np.ndarray = None,
    frame_b: np.ndarray = None,
    window_size: int = 16,
    window_size_steps: int = 3,
    window_size_step_size: int = 2,
    overlap: int = 4,
    overlap_steps: int = 1,
    overlap_step_size: int = 1,
    dt: float = 0.003,
    dt_steps: int = 2,
    dt_step_size: float = 0.0005,
    search_area_size: int = 20,
    search_area_size_steps: int = 1,
    search_area_size_step_size: int = 2,
    sig2noise_method: str = "peak2peak",
    n_threads: int = None
) -> Tuple[DisplacementFromParameters, List[DisplacementFromParameters]]:

    # Create all step lists
    window_sizes = _generate_window(window_size, window_size_steps, window_size_step_size)
    overlaps = _generate_window(overlap, overlap_steps, overlap_step_size)
    dts = _generate_window(dt, dt_steps, dt_step_size)
    search_area_sizes = _generate_window(search_area_size, search_area_size_steps, search_area_size_step_size)

    # Create all permutations
    parameter_keys = ["window_size", "overlap", "dt", "search_area_size"]
    permutations = list(itertools.product(window_sizes, overlaps, dts, search_area_sizes))
    parameter_permutations = [dict(zip(parameter_keys, permutations[i])) for i in range(len(permutations))]

    # Starting multithreading
    with Pool(n_threads) as pool:
        dfps = list(tqdm(pool.imap(
            lambda param_permutation: DisplacementFromParameters(
                calculate_displacement(
                    frames=frames,
                    frame_a=frame_a,
                    frame_b=frame_b,
                    **param_permutation
                ),
                param_permutation
            ),
            parameter_permutations
        ), total=len(parameter_permutations)))

    # Find max
    best = max(dfps, key=lambda dfp: dfp.displacement.median_s2n)

    return best, dfps


def calculate_displacements(frames: List[np.ndarray], n_threads: int = None, **kwargs) -> List[Displacement]:
    # Fence-post: We only want to create displacements for n-frames - 1
    # Additionally enforce frames are numpy types
    frame_pairs = [(frames[i], frames[i+1]) for i in range(len(frames) - 1)]

    # Create partial
    passdown = partial(calculate_displacement, **kwargs)

    # Start multithreading
    with Pool(n_threads) as pool:
        # Map results
        displacements = pool.map(passdown, frame_pairs)

    return displacements
