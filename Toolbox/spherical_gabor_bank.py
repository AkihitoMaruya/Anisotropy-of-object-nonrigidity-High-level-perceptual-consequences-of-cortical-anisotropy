#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spherical 3D Gabor bank: in the frequency domain each filter is an isotropic Gaussian (same SD along f_x, f_y and f_t)
centred at mu = f0 (cos phi cos theta, cos phi sin theta, sin phi). This is the supplementary Eq. S12 Gabor with
sigma_t = sigma_s: the orientation-dependent width (cortical anisotropy) then applies to the temporal direction too.
SD per filter: s_k = f0 * sigma_k (sigma_k = angular width, rad), i.e. the same angular half-width at the centre as the
current banks. Built on GaussianBank (Motion_texture/heeger_gaussian_flow.py) with filters='gaussian', so energies
(one-sided cut) and closed-form white-noise predictions use exactly these Gaussians. New code; nothing existing is changed.
"""

import numpy as np
from anisotropy_heeger import GaussianBank, cat_counts
from gaussian_angle_masks3D import cat_sigma

K16 = 16
THETAS = np.arange(K16)*2*np.pi/K16
S_MEAN = cat_sigma(THETAS).mean()
LAYOUT_C = np.radians([-60, -30, 0, 30, 60])


def widths(beta):
    """Angular widths (rad) per direction: beta = 0 isotropic, 1 cat, 4 = four times the cat anisotropy."""
    return S_MEAN*(1+beta*(cat_sigma(THETAS)/S_MEAN-1))


def spherical_bank(beta, phis=LAYOUT_C, f0=0.12, counts=True, normalize='raw'):
    sig = widths(beta)
    w = cat_counts(THETAS) if (counts and beta > 0) else None
    kw = dict(weights=w, normalize=normalize) if w is not None else {}
    b = GaussianBank(THETAS, sig, phis, f0=f0, sigma_r=f0/3, sigma_t=np.radians(15), filters='gaussian', **kw)
    mu, C = [], []
    for k in range(b.K):
        for j in range(b.S):
            th, ph = THETAS[k], phis[j]
            mu.append(f0*np.array([np.cos(ph)*np.cos(th), np.cos(ph)*np.sin(th), np.sin(ph)]))
            C.append((f0*sig[k])**2*np.eye(3))
    b.mu, b.C = np.array(mu), np.array(C)
    return b


def current_bank(beta, phis=LAYOUT_C, f0=0.12, counts=True, normalize='raw'):
    """The bank used so far (exact product filters for energies, linearised Gaussians for predictions)."""
    sig = widths(beta)
    w = cat_counts(THETAS) if (counts and beta > 0) else None
    kw = dict(weights=w, normalize=normalize) if w is not None else {}
    return GaussianBank(THETAS, sig, phis, f0=f0, sigma_r=f0/3, sigma_t=np.radians(15), **kw)


# ---------------------------------------------------------------------------
# Widths from supplementary Eq. S14: Delta_omega_1/2 = 2 atan(0.5 / (F0 sigma)), with Delta_omega_1/2 set to the cat
# half-height half-width h. sigma (px) is the spatial Gabor SD (= temporal SD, spherical); frequency SD = 1/(2 pi sigma).
# ---------------------------------------------------------------------------
def cat_hwhh(beta):
    """Half-height half-widths (rad) per direction, scaled by beta around the mean (beta 0 isotropic, 1 cat)."""
    h = widths(1)*np.sqrt(2*np.log(2))
    hm = h.mean()
    return hm*(1+beta*(h/hm-1))


def s14_sigma(beta, f0=0.12):
    """Spatial (and temporal) Gabor SD in px per direction from Eq. S14."""
    return 0.5/(f0*np.tan(cat_hwhh(beta)/2))


def s14_bank(beta, phis=LAYOUT_C, f0=0.12, counts=True, normalize='raw'):
    s = 1/(2*np.pi*s14_sigma(beta, f0))                  # frequency-domain SD, cycles/sample
    w = cat_counts(THETAS) if (counts and beta > 0) else None
    kw = dict(weights=w, normalize=normalize) if w is not None else {}
    b = GaussianBank(THETAS, widths(beta), phis, f0=f0, sigma_r=f0/3, sigma_t=np.radians(15), filters='gaussian', **kw)
    mu, C = [], []
    for k in range(b.K):
        for j in range(b.S):
            th, ph = THETAS[k], phis[j]
            mu.append(f0*np.array([np.cos(ph)*np.cos(th), np.cos(ph)*np.sin(th), np.sin(ph)]))
            C.append(s[k]**2*np.eye(3))
    b.mu, b.C = np.array(mu), np.array(C)
    return b


# ---------------------------------------------------------------------------
# Exact match: SD chosen so that the filter's half-height half-width in orientation (response to a grating at the
# preferred spatial frequency F0, rotated by Delta) equals the cat value h exactly:
#     exp(-(2 F0 sin(h/2))^2 / (2 s^2)) = 1/2   ->   s = 2 F0 sin(h/2) / sqrt(2 ln 2);  spatial Gabor SD = 1/(2 pi s).
# ---------------------------------------------------------------------------
def exact_sd(beta, f0=0.12):
    return 2*f0*np.sin(cat_hwhh(beta)/2)/np.sqrt(2*np.log(2))


def exact_bank(beta, phis=LAYOUT_C, f0=0.12, counts=True, normalize='raw'):
    s = exact_sd(beta, f0)
    w = cat_counts(THETAS) if (counts and beta > 0) else None
    kw = dict(weights=w, normalize=normalize) if w is not None else {}
    b = GaussianBank(THETAS, widths(beta), phis, f0=f0, sigma_r=f0/3, sigma_t=np.radians(15), filters='gaussian', **kw)
    mu, C = [], []
    for k in range(b.K):
        for j in range(b.S):
            th, ph = THETAS[k], phis[j]
            mu.append(f0*np.array([np.cos(ph)*np.cos(th), np.cos(ph)*np.sin(th), np.sin(ph)]))
            C.append(s[k]**2*np.eye(3))
    b.mu, b.C = np.array(mu), np.array(C)
    return b


def exact_bank_isotropic_time(beta, phis=LAYOUT_C, f0=0.12, counts=True, normalize='raw'):
    """Spatial SD per direction matched exactly to the cat half-widths (as exact_bank); temporal SD the same for all
    filters (the isotropic value): C = diag(s_k^2, s_k^2, s_iso^2) in (f_x, f_y, f_t)."""
    s = exact_sd(beta, f0)
    st = exact_sd(0, f0)[0]
    w = cat_counts(THETAS) if (counts and beta > 0) else None
    kw = dict(weights=w, normalize=normalize) if w is not None else {}
    b = GaussianBank(THETAS, widths(beta), phis, f0=f0, sigma_r=f0/3, sigma_t=np.radians(15), filters='gaussian', **kw)
    mu, C = [], []
    for k in range(b.K):
        for j in range(b.S):
            th, ph = THETAS[k], phis[j]
            mu.append(f0*np.array([np.cos(ph)*np.cos(th), np.cos(ph)*np.sin(th), np.sin(ph)]))
            C.append(np.diag([s[k]**2, s[k]**2, st**2]))
    b.mu, b.C = np.array(mu), np.array(C)
    return b


# ---------------------------------------------------------------------------
# Motion-energy match: the half-height half-width h is that of the ENERGY response (squared quadrature-pair output,
# a complex-cell-like unit), not of the linear (simple-cell, odd or even) amplitude:
#     exp(-(2 F0 sin(h/2))^2 / s^2) = 1/2   ->   s = 2 F0 sin(h/2) / sqrt(ln 2)   (= sqrt(2) x exact_sd)
# ---------------------------------------------------------------------------
def me_sd(beta, f0=0.12):
    return 2*f0*np.sin(cat_hwhh(beta)/2)/np.sqrt(np.log(2))


def me_bank(beta, phis=LAYOUT_C, f0=0.12, counts=True, normalize='raw'):
    s = me_sd(beta, f0)
    w = cat_counts(THETAS) if (counts and beta > 0) else None
    kw = dict(weights=w, normalize=normalize) if w is not None else {}
    b = GaussianBank(THETAS, widths(beta), phis, f0=f0, sigma_r=f0/3, sigma_t=np.radians(15), filters='gaussian', **kw)
    mu, C = [], []
    for k in range(b.K):
        for j in range(b.S):
            th, ph = THETAS[k], phis[j]
            mu.append(f0*np.array([np.cos(ph)*np.cos(th), np.cos(ph)*np.sin(th), np.sin(ph)]))
            C.append(s[k]**2*np.eye(3))
    b.mu, b.C = np.array(mu), np.array(C)
    return b


# ---------------------------------------------------------------------------
# Mirror-symmetric (folded) cat data: widths and cell numbers averaged over theta and 180 - theta, so that mirror-image
# orientations (e.g. 22.5 and 157.5 deg) are treated identically; the cardinal-oblique differences are kept.
# ---------------------------------------------------------------------------
MIRROR = (8-np.arange(K16)) % K16                                   # index of 180 - theta


def fold(x):
    return (np.asarray(x)+np.asarray(x)[MIRROR])/2


def exact_bank_sym(beta, phis=LAYOUT_C, f0=0.12, counts=True, normalize='raw'):
    """exact_bank with folded half-widths and folded cell numbers."""
    h = fold(cat_hwhh(beta))
    s = 2*f0*np.sin(h/2)/np.sqrt(2*np.log(2))
    w = fold(cat_counts(THETAS)) if (counts and beta > 0) else None
    kw = dict(weights=w, normalize=normalize) if w is not None else {}
    b = GaussianBank(THETAS, fold(widths(beta)), phis, f0=f0, sigma_r=f0/3, sigma_t=np.radians(15), filters='gaussian', **kw)
    mu, C = [], []
    for k in range(b.K):
        for j in range(b.S):
            th, ph = THETAS[k], phis[j]
            mu.append(f0*np.array([np.cos(ph)*np.cos(th), np.cos(ph)*np.sin(th), np.sin(ph)]))
            C.append(s[k]**2*np.eye(3))
    b.mu, b.C = np.array(mu), np.array(C)
    return b
