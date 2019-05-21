#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame

npvec = np.ndarray
npgrid = np.ndarray


class Displacement(object):

    def __init__(
        self,
        u: npvec,
        v: npvec,
        sig2noise: npvec,
        x: npvec = None,
        y: npvec = None,
        mask: npvec = None
    ):
        self._x = x
        self._y = y
        self._mask = mask
        self._u = u
        self._v = v
        self._sig2noise = sig2noise
        self._m = None
        self._m_grid = None

        # Lazy load
        self._median_s2n = None

    @property
    def x(self) -> npvec:
        return self._x

    @property
    def y(self) -> npvec:
        return self._y

    @property
    def u(self) -> npvec:
        return self._u

    @property
    def v(self) -> npvec:
        return self._v

    @property
    def mask(self) -> npvec:
        return self._mask

    @property
    def magnitude(self) -> npvec:
        if self._m is None:
            self._m = np.sqrt(self._u ** 2 + self._v ** 2)
        return self._m

    @property
    def sig2noise(self):
        return self._sig2noise

    @property
    def median_s2n(self):
        if self._median_s2n is None:
            self._median_s2n = np.median(self.sig2noise.flatten(), axis=0)

        return self._median_s2n

    @property
    def magnitude_grid(self) -> npgrid:
        if self._m_grid is None:
            df = DataFrame({'x': self.x.flatten(), 'y': self.y.flatten(), 'm': self.magnitude.flatten()})
            self._m_grid = df.pivot('y', 'x', 'm').values
        return self._m_grid

    def __str__(self):
        output = "<Displacement ["
        if self._x is not None:
            output += f"x:{self._x.shape}, "
        if self._y is not None:
            output += f"y:{self._y.shape}, "
        if self._u is not None:
            output += f"u:{self._u.shape}, "
        output += f"sig2noise:{self.median_s2n}]>"
        return output

    def __repr__(self):
        return str(self)
