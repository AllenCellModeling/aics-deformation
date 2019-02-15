#!/usr/bin/env python
# -*- coding: utf-8 -*-

from aicsdeformation import Deformation


def test_value_change():
    start_val = 5
    new_val = 20
    #
    example = Deformation(start_val)
    example.update_value(new_val)
    assert (example.get_value() == new_val and
            example.get_previous_value() == start_val)
