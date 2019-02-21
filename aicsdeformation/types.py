#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import NamedTuple


class Displacement(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    mask: np.ndarray
    sig2noise: np.ndarray
