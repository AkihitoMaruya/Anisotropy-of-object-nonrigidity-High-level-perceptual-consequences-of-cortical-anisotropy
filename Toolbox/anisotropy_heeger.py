#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anisotropy-project wrapper around the Gaussian Heeger model, whose canonical code lives in the
lab Motion_texture folder (heeger_gaussian_flow.py, derivation in heeger_gaussian_model.md).
Adds the cat V1 cell counts per direction.
"""

import sys
import numpy as np

from heeger_gaussian_flow import GaussianBank, derived_gaussian  # noqa: E402,F401

# Cat V1 cells per motion direction (Toolbox/Data/anisotropy_pyr_num_cells_dir.npy),
# directions -180, -157.5, ..., 157.5 deg; 180-deg periodic.
_NDIR = np.array([179., 140.5, 144., 146., 209.5, 170.5, 152., 165.])


def cat_counts(theta):
    i = np.round((np.degrees(np.asarray(theta))+180)/22.5).astype(int) % 8
    return _NDIR[i]/_NDIR.mean()
