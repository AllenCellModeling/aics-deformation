#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np
from typing import List, Tuple

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# openpiv reference:
# https://openpiv.readthedocs.io/en/latest/src/tutorial.html


class AICSDeformation(object):
    """
    AICSDeformation objects can be used to process and visualize standard deformation tasks.

    :param frames: A list of imaging frames to compare.
    """

    @staticmethod
    def _check_frames(frames: List[np.ndarray], dims: Tuple[int, int]):
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

    def __init__(self, frames: List[np.ndarray]):
        # Check frames
        self._check_frames(frames, frames[0].shape)

        # Store frames
        self._frames = copy.deepcopy(frames)

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
            self._frames.insert(index + i, copy.deepcopy(frame))

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
        self._frames = [*self.frames, *[copy.deepcopy(frame) for frame in frames]]

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
