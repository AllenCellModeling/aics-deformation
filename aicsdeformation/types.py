#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Displacement(object):

    def __init__(
        self,
        u: np.ndarray,
        v: np.ndarray,
        sig2noise: np.ndarray,
        x: np.ndarray = None,
        y: np.ndarray = None,
        mask: np.ndarray = None
    ):
        self._x = x
        self._y = y
        self._mask = mask
        self._u = u
        self._v = v
        self._sig2noise = sig2noise

        # Lazy load
        self._median_s2n = None

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def u(self):
        return self._u

    @property
    def v(self):
        return self._v

    @property
    def mask(self):
        return self._mask

    @property
    def sig2noise(self):
        return self._sig2noise

    @property
    def median_s2n(self):
        if self._median_s2n is None:
            self._median_s2n = np.median(self.sig2noise.flatten(), axis=0)

        return self._median_s2n

    def __str__(self):
        return f"<Displacement [sig2noise: {self.median_s2n}]>"

    def __repr__(self):
        return str(self)
