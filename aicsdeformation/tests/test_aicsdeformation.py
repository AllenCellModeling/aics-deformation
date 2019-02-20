#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from aicsdeformation import AICSDeformation

SINGLE_FRAME = np.ones((2, 2))
VALID_FRAMES = [SINGLE_FRAME for i in range(2)]
TYPE_ERROR_FRAMES = [*VALID_FRAMES, "hello world"]
SHAPE_ERROR_FRAMES = [*VALID_FRAMES, np.ones((2, 2, 2))]
DIMS_ERROR_FRAMES = [*VALID_FRAMES, np.ones((1, 3))]


@pytest.mark.parametrize("frames, expected_frames, expected_len, expected_repr", [
    (VALID_FRAMES, VALID_FRAMES, 2, f"<AICSDeformation [2 x (2, 2) frames]>"),
    # Test types
    pytest.param(
        TYPE_ERROR_FRAMES,
        None,
        None,
        None,
        marks=pytest.mark.raises(exception=TypeError)
    ),
    # Test shapes
    pytest.param(
        SHAPE_ERROR_FRAMES,
        None,
        None,
        None,
        marks=pytest.mark.raises(exception=ValueError)
    ),
    # Test dims
    pytest.param(
        DIMS_ERROR_FRAMES,
        None,
        None,
        None,
        marks=pytest.mark.raises(exception=ValueError)
    )
])
def test_aicsdeformation_init(frames, expected_frames, expected_len, expected_repr):
    """
    This can throw multiple errors before reaching the assertion due to frame checking
    internal to AICSDeformation.
    """
    actual = AICSDeformation(frames)
    assert all(np.array_equal(actual[i], expected_frames[i]) for i in range(len(expected_frames)))
    assert len(actual) == expected_len
    assert str(actual) == expected_repr


@pytest.mark.parametrize("frames, index, new_frames, expected_frames", [
    (VALID_FRAMES, 0, [SINGLE_FRAME * 2, SINGLE_FRAME * 3], [SINGLE_FRAME * 2, SINGLE_FRAME * 3, *VALID_FRAMES]),
    (VALID_FRAMES, 1, [SINGLE_FRAME * 2, SINGLE_FRAME * 3],
        [SINGLE_FRAME, SINGLE_FRAME * 2, SINGLE_FRAME * 3, SINGLE_FRAME]),
    pytest.param(
        VALID_FRAMES,
        5,
        [SINGLE_FRAME * 2],
        None,
        marks=pytest.mark.raises(exception=IndexError)
    )
])
def test_aicsdeformation_insert_frames(frames, index, new_frames, expected_frames):
    actual = AICSDeformation(frames)
    actual.insert_frames(index, new_frames)
    assert all(np.array_equal(actual[i], expected_frames[i]) for i in range(len(expected_frames)))


@pytest.mark.parametrize("frames, index, frame, expected_frames", [
    (VALID_FRAMES, 0, SINGLE_FRAME * 2, [SINGLE_FRAME * 2, *VALID_FRAMES]),
    (VALID_FRAMES, 1, SINGLE_FRAME * 2, [SINGLE_FRAME, SINGLE_FRAME * 2, SINGLE_FRAME]),
    pytest.param(
        VALID_FRAMES,
        5,
        SINGLE_FRAME * 2,
        None,
        marks=pytest.mark.raises(exception=IndexError)
    )
])
def test_aicsdeformation_insert_frame(frames, index, frame, expected_frames):
    actual = AICSDeformation(frames)
    actual.insert_frame(index, frame)
    assert all(np.array_equal(actual[i], expected_frames[i]) for i in range(len(expected_frames)))


@pytest.mark.parametrize("frames, new_frames, expected_frames", [
    (VALID_FRAMES, [SINGLE_FRAME * 2, SINGLE_FRAME * 4], [*VALID_FRAMES, SINGLE_FRAME * 2, SINGLE_FRAME * 4]),
    (VALID_FRAMES, [], VALID_FRAMES)
])
def test_aicsdeformation_append_frames(frames, new_frames, expected_frames):
    actual = AICSDeformation(frames)
    actual.append_frames(new_frames)
    assert all(np.array_equal(actual[i], expected_frames[i]) for i in range(len(expected_frames)))


@pytest.mark.parametrize("frames, frame, expected_frames", [
    (VALID_FRAMES, SINGLE_FRAME * 2, [*VALID_FRAMES, SINGLE_FRAME * 2]),
    (VALID_FRAMES, SINGLE_FRAME * 4, [*VALID_FRAMES, SINGLE_FRAME * 4])
])
def test_aicsdeformation_append_frame(frames, frame, expected_frames):
    actual = AICSDeformation(frames)
    actual.append_frame(frame)
    assert all(np.array_equal(actual[i], expected_frames[i]) for i in range(len(expected_frames)))


@pytest.mark.parametrize("frames, index, new, expected_old, expected_frames", [
    (VALID_FRAMES, 0, SINGLE_FRAME * 2, SINGLE_FRAME, [SINGLE_FRAME * 2, SINGLE_FRAME]),
    (VALID_FRAMES, 1, SINGLE_FRAME * 2, SINGLE_FRAME, [SINGLE_FRAME, SINGLE_FRAME * 2]),
    pytest.param(
        VALID_FRAMES,
        5,
        SINGLE_FRAME,
        None,
        None,
        marks=pytest.mark.raises(exception=IndexError)
    )
])
def test_aicsdeformation_update_frame(frames, index, new, expected_old, expected_frames):
    actual = AICSDeformation(frames)
    old = actual.update_frame(index, new)
    assert np.array_equal(old, expected_old)
    assert all(np.array_equal(actual[i], expected_frames[i]) for i in range(len(expected_frames)))
