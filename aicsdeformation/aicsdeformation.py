#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np
from typing import List, Tuple, Union

from .processing import calculate_displacements, process_displacements

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class AICSDeformation(object):
    """
    AICSDeformation objects can be used to process and visualize standard deformation tasks.

    :param frames: An ordered list of imaging frames, or a 3D ndarray, to compare.
    :param t_index: If frames was provided as a 3D ndarray, which dimension index represents time.
    """

    @staticmethod
    def _check_frames(frames: List[np.ndarray], dims: Tuple[int, int]):
        log.debug(f"Validating {len(frames)} frames using dims: {dims}.")
        for frame in frames:
            # Check frame is a numpy.ndarray
            if not isinstance(frame, np.ndarray):
                raise TypeError("Frame data must be a numpy.ndarray.")

            # Check frame is a 2D numpy.ndarray
            if len(frame.shape) != 2:
                raise ValueError("Frame data must be a 2D numpy.ndarray.")

            # Check frame dimensions match
            if frame.shape != dims:
                raise ValueError("All frame data must have the same dimensions.")

    def __init__(self, frames: Union[np.ndarray, List[np.ndarray]], t_index: int = None):
        # Convert 3D ndarray
        if isinstance(frames, np.ndarray):
            # Check dim size
            if len(frames.shape) != 3:
                raise ValueError(f"Unsure how to handle non 3D frames ndarray. Provided: {len(frames.shape)}D")

            # Check t_index
            if t_index is None:
                raise TypeError("No t_index was provided to split 3D ndarray into frames.")

            # Split
            frames = [frames[t, :, :] for t in range(frames.shape[t_index])]

        # Check frames
        self._check_frames(frames, frames[0].shape)

        # Store frames
        self._frames = [copy.deepcopy(frame.astype(np.uint16)) for frame in frames]

        # Lazy load
        self._displacements = None

    @property
    def frames(self):
        return self._frames

    @property
    def dims(self):
        return self[0].shape

    def insert_frames(self, index: int, frames: List[np.ndarray]):
        """
        Insert multiple frames.

        :param index: Which index to insert the frames at.
        :param frames: List of frames to insert.
        """
        # Check frames
        self._check_frames(frames, self.dims)

        # Check index
        if index >= len(self):
            raise IndexError(f"Index provided {index} is out of bounds of current frame list.")

        # Insert all
        for i, frame in enumerate(frames):
            self._frames.insert(index + i, copy.deepcopy(frame.astype(np.uint16)))
            log.info(f"Inserted {frame.shape} frame at {i}.")

    def insert_frame(self, index: int, frame: np.ndarray):
        """
        Insert a single frame.

        :param index: Which frame index to insert a new frame at.
        :param frame: New frame data to set at the index.
        """
        # Use insert_frames
        self.insert_frames(index, [frame])

    def append_frames(self, frames: List[np.ndarray]):
        """
        Append multiple frames.

        :param frames: List of frames to append.
        """
        # Check frames
        self._check_frames(frames, self.dims)

        # Append frames
        self._frames = [*self.frames, *[copy.deepcopy(frame.astype(np.uint16)) for frame in frames]]
        log.info(f"Appended {len(frames)} to frames.")

    def append_frame(self, frame: np.ndarray):
        """
        Append a single frame.

        :param frame: New frame data to append.
        """
        # Use append_frames
        self.append_frames([frame])

    def remove_frame(self, index):
        """
        Remove a single frame.

        :param index: Which frame index to remove.
        :return: The previously stored frame data at the provided index.
        """
        log.info(f"Removing frame at {index}.")
        return self._frames.pop(index)

    def update_frame(self, index: int, frame: np.ndarray):
        """
        Update a single frame.

        :param index: Which frame index to update.
        :param frame: New frame data to set at the index, if None, the frame at target index will be removed.
        :return: The previously stored frame data at the provided index.
        """
        # Check frame
        self._check_frames([frame], self.dims)

        # Insert and remove old
        self.insert_frame(index, frame)
        return self.remove_frame(index + 1)

    @property
    def displacements(self):
        if self._displacements is None:
            self._displacements = calculate_displacements(self)

        return self._displacements

    def generate_displacements(
        self,
        window_size: int = 18,
        overlap: int = 4,
        dt: float = 0.003,
        search_area_size: int = 20,
        sig2noise_method: str = "peak2peak",
        s2n_threshold: float = 1.3,
        outlier_method: str = "localmean",
        max_iter: int = 10,
        kernal_size: int = 2,
        scaling_factor: float = 96.52,
        n_threads: int = None
    ):
        """
        Generate displacement objects for all frames of present in the AICSDeformation object.
        At the completion of this function, these displacements will be available from the `displacements` attribute.

        Look to OpenPIV for details on how parameters interact with the actual displacement generation.
        https://openpiv.readthedocs.io/en/latest/src/tutorial.html

        Additionally, a grid search implementation is available in the case you want to have aicsdeformation
        find the the best signal : noise given starting parameters, example below.

        ```
        from aicsdeformation.processing import grid_search_displacements
        best, all_displacements = grid_search_displacements(frame_a, frame_b)
        ```
        """
        # Get base
        self._displacements = calculate_displacements(
            self[:2],
            window_size=window_size,
            overlap=overlap,
            dt=dt,
            search_area_size=search_area_size,
            sig2noise_method=sig2noise_method,
            n_threads=n_threads
        )

        # Process
        self._displacements = process_displacements(
            self.displacements,
            image_size=self.dims,
            window_size=window_size,
            overlap=overlap,
            s2n_threshold=s2n_threshold,
            outlier_method=outlier_method,
            max_iter=max_iter,
            kernal_size=kernal_size,
            scaling_factor=scaling_factor,
            n_threads=n_threads
        )

        return self._displacements

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]

    def __iter__(self):
        yield from self.frames

    def __str__(self):
        return f"<AICSDeformation [{len(self)} x {self.dims} frames]>"

    def __repr__(self):
        return str(self)
