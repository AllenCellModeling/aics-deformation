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
    parameters: Dict[str, Union[int, float, str]]


class ErrorFromParameters(NamedTuple):
    error: ValueError
    parameters: Dict[str, Union[int, float, str]]


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
    """
    Generate base displacement. This is primarly used for determining signal : noise between two frames before
    additional processing is conducted.

    :param frames: A tuple of two numpy.ndarray frames to find displacements for.
        This will be expanded to frame_a and frame_b if not None.
    :param frame_a: A single frame to be used to find displacements for.
    :param frame_b: A single frame to be used to find displacements for.

    Look to OpenPIV for details on how parameters interact with the actual displacement generation.
    https://openpiv.readthedocs.io/en/latest/src/tutorial.html
    """
    # Expand frames if required
    if frames:
        frame_a, frame_b = frames

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
    return Displacement(u=u, v=v, sig2noise=sig2noise)


def calculate_displacements(frames: List[np.ndarray], n_threads: int = None, **kwargs) -> List[Displacement]:
    """
    Wrapper around calculate_displacement to multithread calculate multiple frame pairs at the same time.

    :param frames: An ordered list of numpy.ndarray frames to create base Displacement objects for.
        Displacements will be created for n-1 frames where n is the length of the frames list.
    :param n_threads: How many threads should be used for this operation. If None, os.cpu_count() is used.
    :param **kwargs: Keyword arguments passed down to the calculate_displacement function.
    :return: The completed list of generated base Displacement objects.
    """
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


def _generate_window(center: Union[int, float], steps: int, step_size: Union[int, float]) -> Set[Union[int, float]]:
    window = set([
        *[center - i * step_size for i in range(steps + 1)],
        *[center + i * step_size for i in range(steps + 1)]
    ])

    return window


def _grid_search_lambda(frames, frame_a, frame_b, param_permutation) -> DisplacementFromParameters:
    # Try except function to be used as lambda from grid search
    try:
        return DisplacementFromParameters(
            calculate_displacement(
                frames=frames,
                frame_a=frame_a,
                frame_b=frame_b,
                **param_permutation
            ),
            param_permutation
        )
    except ValueError as e:
        return ErrorFromParameters(e, param_permutation)


def grid_search_displacements(
    frames: Tuple[np.ndarray] = None,
    frame_a: np.ndarray = None,
    frame_b: np.ndarray = None,
    window_size: int = 16,
    window_size_steps: int = 7,
    window_size_step_size: int = 2,
    overlap: int = 4,
    overlap_steps: int = 3,
    overlap_step_size: int = 1,
    dt: float = 0.003,
    dt_steps: int = 5,
    dt_step_size: float = 0.0005,
    search_area_size: int = 20,
    search_area_size_steps: int = 9,
    search_area_size_step_size: int = 2,
    sig2noise_method: str = "peak2peak",
    n_threads: int = None
) -> Tuple[DisplacementFromParameters, List[DisplacementFromParameters]]:
    """
    Grid search for parameters that result in the highest signal : noise.

    This will not process the displacements fully and will only return parameter information and an associated
    incompletely processed displacement. The best/ highest signal : noise displacement is returned, along with all
    displacements created during the grid search process. These are stored not as Displacement objects but as
    DisplacementFromParameters, which from the name, is meerly an abstraction to store the Displacement object with the
    parameters used to create it.

    :param frames: A tuple of two numpy.ndarray frames to find displacements for.
        This will be expanded to frame_a and frame_b if not None.
    :param frame_a: A single frame to be used to find displacements for.
    :param frame_b: A single frame to be used to find displacements for.
    :param n_threads: How many threads should be used for the grid search operation. If None, os.cpu_count() is used.
    :param *steps: Any parameter that ends in 'steps' is the number of steps away from the origin of the associated
        parameter you want to search. In the case where window_size is 16 and window_size_steps is 2, this will create
        displacements for the parameter window_size:
            [16 - step_size * 2, 16 - step_size, 16, 16 + step_size, 16 + step_size * 2]
    :param *step_size: Any parameter that ends in 'step_size' is the value for how large each step should be from the
        origin. In the case where window_size is 16, window_size_steps is 2, and window_size_step_size is 1, this will
        creat displacements for the parameter window_size: [14, 15, 16, 17, 18]
    :return: Tuple of the best DisplacementFromParameters and a list of all DisplacementFromParameters searched. Best
        is the DisplacementFromParameters with the highest median signal : noise out of all DisplacementFromParameters.

    Look to OpenPIV for details on how parameters interact with the actual displacement generation.
    https://openpiv.readthedocs.io/en/latest/src/tutorial.html
    """
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
            lambda param_permutation: _grid_search_lambda(
                frames=frames,
                frame_a=frame_a,
                frame_b=frame_b,
                param_permutation=param_permutation
            ),
            parameter_permutations
        ), total=len(parameter_permutations)))

    # Find max
    best_dfp = None
    best_s2n = -1
    for dfp in dfps:
        try:
            if dfp.displacement.median_s2n > best_s2n:
                best_dfp = dfp
                best_s2n = dfp.displacement.median_s2n
        except AttributeError:
            pass

    return best_dfp, dfps


def process_displacement(
    d: Displacement,
    image_size: Tuple[int],
    window_size: int = 16,
    overlap: int = 4,
    s2n_threshold: float = 1.3,
    outlier_method: str = "localmean",
    max_iter: int = 10,
    kernal_size: int = 2,
    scaling_factor: float = 96.52
) -> Displacement:
    """
    Process an already created base displacement and return a fully filled Displacement object.

    :param d: A base Displacement object created by calculate_displacement.
    :param image_size: The size of one of the frames used to create d. This is usually, the frame.shape.

    Look to OpenPIV for details on how parameters interact with the actual displacement generation.
    https://openpiv.readthedocs.io/en/latest/src/tutorial.html
    """
    # Generate x, y
    x, y = openpiv.process.get_coordinates(image_size=image_size, window_size=window_size, overlap=overlap)

    # Validate signal : noise
    u, v, mask = openpiv.validation.sig2noise_val(d.u, d.v, d.sig2noise, threshold=s2n_threshold)

    # Replace outliers
    u, v = openpiv.filters.replace_outliers(u, v, method=outlier_method, max_iter=max_iter, kernel_size=kernal_size)

    # Scale
    x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor=scaling_factor)

    # Create and return new displacement
    return Displacement(x=x, y=y, u=d.u, v=d.v, mask=mask, sig2noise=d.sig2noise)


def process_displacements(displacements: List[Displacement], n_threads: int = None, **kwargs) -> List[Displacement]:
    """
    Wrapper around process_displacement to multithread process multiple displacements at the same time.

    :param displacements: A list of Displacements to finish processing.
    :param n_threads: How many threads should be used for this operation. If None, os.cpu_count() is used.
    :param **kwargs: Keyword arguments passed down to the process_displacement function.
    :return: The completed list of generated base Displacement objects.
    """
    # Create partial
    passdown = partial(process_displacement, **kwargs)

    # Start multithreading
    with Pool(n_threads) as pool:
        # Map results
        displacements = pool.map(passdown, displacements)

    return displacements
