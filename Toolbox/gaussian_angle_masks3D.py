#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian angle masks for 3D (x, y, t) motion filters, defined in the Fourier domain.

Each filter = radial Gaussian x spatial angle Gaussian x temporal angle Gaussian, i.e. a Gaussian
in spherical frequency coordinates (|f|, theta, a), on the same frequency grid as
steerable_pyramid3D.py (2026_motion_text_sude):

  radial    R(f)    = exp(-(|f| - f0)^2 / (2 sigma_r^2)),  |f| = sqrt(fx^2 + fy^2 + ft^2)

  spatial   G_b(f)  = exp(-d_b^2 / (2 sigma_b^2)) for |d_b| < pi/2, else 0
            d_b = wrapped angle of (fx, fy) minus theta_b.
            One-sided (a single lobe around theta_b, nothing around theta_b + pi), so the
            inverse FFT is an analytic filter: real part = even, imaginary part = odd
            (the spatial quadrature pair). sigma_b can differ between orientations.
  temporal  T_t(f)  = exp(-(a_b - phi_t)^2 / (2 sigma_t^2)),
            a_b = atan2(ft, cos(theta_b) fx + sin(theta_b) fy) in (-pi/2, pi/2) on the
            one-sided half; phi_t sets speed and direction along theta_b
            (phi_t = 0 static, sign of phi_t = direction of motion).

theta is the orientation of the spatial-frequency vector, i.e. the direction normal to the
preferred edge (theta = 0: vertical edges, horizontal motion).
"""

import numpy as np

# Cat V1 half-width at half-height (deg) by preferred EDGE orientation, -180..157.5 in 22.5 deg
# steps (Li, Peterson & Freeman 2003; values used in Figure3.py), averaged over opposite directions.
_TW = np.array([28.138, 29.793, 34.759, 38.069, 35.586, 40.552, 33.931, 30.207,
                28.552, 32.276, 34.759, 38.483, 35.586, 37.655, 33.931, 32.276])
_TW = (_TW[:8]+_TW[8:])/2
_TW_ORI = np.radians(np.arange(-180, 0, 22.5))          # edge orientations of _TW (mod pi)


def cat_sigma(theta):
    """Anisotropic spatial angular sigma (rad) for frequency orientation theta (rad).
    The frequency vector is normal to the edge, so edge orientation = theta - pi/2.
    HWHH -> Gaussian sigma: sigma = HWHH / sqrt(2 ln 2)."""
    edge = np.mod(np.asarray(theta)-np.pi/2, np.pi)
    xp = np.mod(_TW_ORI, np.pi)
    order = np.argsort(xp)
    hwhh = np.interp(edge, xp[order], _TW[order], period=np.pi)
    return np.radians(hwhh)/np.sqrt(2*np.log(2))


def freq_grid(shape):
    """fftshifted frequency grid in rad/sample, same convention as steerable_pyramid3D."""
    r = [np.linspace(-np.pi, np.pi, n+1)[:-1] for n in shape]
    return np.meshgrid(*r, indexing='ij')                  # FT, FY, FX


def _log_raised_cosine_hi(r):
    h = np.where(np.abs(r) >= np.pi/2, 1.0, 0.0)
    m = (np.abs(r) > np.pi/4) & (np.abs(r) < np.pi/2)
    h[m] = np.cos(np.pi/2*np.log2(np.abs(r[m])*2/np.pi))
    return h


def _log_raised_cosine_lo(r):
    lo = np.where(np.abs(r) <= np.pi/4, 1.0, 0.0)
    m = (np.abs(r) > np.pi/4) & (np.abs(r) < np.pi/2)
    lo[m] = np.cos(np.pi/2*np.log2(np.abs(r[m])*4/np.pi))
    return lo


def radial_band(FT, FY, FX, scale=0):
    """Band-pass of pyramid scale `scale` (octave-wide log raised cosine), undecimated."""
    R = np.sqrt(FX**2+FY**2+FT**2)/2*2**scale
    return _log_raised_cosine_lo(R)*_log_raised_cosine_hi(2*R)


def radial_gaussian(FT, FY, FX, f0=0.25, sigma_r=0.075):
    """Gaussian in 3D frequency magnitude; f0 and sigma_r in cycles/sample."""
    rho = np.sqrt(FX**2+FY**2+FT**2)/(2*np.pi)
    return np.exp(-(rho-f0)**2/(2*sigma_r**2))


def _wrap(a):
    return (a+np.pi) % (2*np.pi)-np.pi


def spatial_angle_mask(FY, FX, theta, sigma, one_sided=True):
    ang = np.arctan2(FY, FX)
    d = _wrap(ang-theta)
    G = np.exp(-d**2/(2*sigma**2))*(np.abs(d) < np.pi/2)
    G[(FX == 0) & (FY == 0)] = 0                           # no spatial frequency -> no orientation
    if one_sided:
        return G
    d2 = _wrap(ang-theta-np.pi)
    return G+np.exp(-d2**2/(2*sigma**2))*(np.abs(d2) < np.pi/2)


def temporal_angle_mask(FT, FY, FX, theta, phi, sigma_t):
    proj = np.cos(theta)*FX+np.sin(theta)*FY
    a = np.arctan2(FT, proj)                               # in (-pi/2, pi/2) where proj > 0
    return np.exp(-_wrap(a-phi)**2/(2*sigma_t**2))


def make_masks(shape, thetas, sigmas, phis, sigma_t, f0=0.25, sigma_r=0.075, one_sided=True):
    """Filter masks, array (K, S, T, H, W).
    thetas: K spatial directions (rad); sigmas: scalar, length-K array, or callable theta -> sigma;
    phis: S temporal angles (rad); sigma_t: temporal angular sigma (rad);
    f0, sigma_r: radial Gaussian centre and width (cycles/sample)."""
    FT, FY, FX = freq_grid(shape)
    thetas = np.atleast_1d(thetas)
    if callable(sigmas):
        sig = np.array([sigmas(th) for th in thetas])
    else:
        sig = np.broadcast_to(np.asarray(sigmas, float), thetas.shape)
    rad = radial_gaussian(FT, FY, FX, f0, sigma_r)
    masks = np.zeros((len(thetas), len(phis))+tuple(shape))
    for k, (th, s) in enumerate(zip(thetas, sig)):
        G = spatial_angle_mask(FY, FX, th, s, one_sided)
        for j, ph in enumerate(phis):
            masks[k, j] = rad*G*temporal_angle_mask(FT, FY, FX, th, ph, sigma_t)
    return masks


def kernel(mask):
    """Space-time filter (complex) from an fftshifted mask: Re = even, Im = odd if one-sided."""
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(mask)))


def gaussian_fit_r2(mask, shape):
    """How close a mask is to a single 3D Gaussian in Cartesian frequency: build the Gaussian
    with the mask's own mean and covariance (mask treated as a distribution) and return
    the R^2 between the two."""
    FT, FY, FX = freq_grid(shape)
    X = np.stack([FT.ravel(), FY.ravel(), FX.ravel()], 1)
    w = mask.ravel()/mask.sum()
    mu = w@X
    C = (X-mu).T@((X-mu)*w[:, None])
    d = X-mu
    g = np.exp(-0.5*np.einsum('ij,jk,ik->i', d, np.linalg.inv(C), d))
    g *= (mask.ravel()@g)/(g@g)
    return 1-np.sum((mask.ravel()-g)**2)/np.sum((mask.ravel()-mask.mean())**2)


# ---------------------------------------------------------------------------
# Cartesian 3D Gaussian filters (moment-matched to the product masks above)
# ---------------------------------------------------------------------------
def fit_gaussian(mask, shape):
    """Mean mu and covariance C of a mask treated as a distribution over (ft, fy, fx),
    in cycles/sample."""
    FT, FY, FX = freq_grid(shape)
    X = np.stack([FT.ravel(), FY.ravel(), FX.ravel()], 1)/(2*np.pi)
    w = mask.ravel()/mask.sum()
    mu = w@X
    d = X-mu
    return mu, d.T@(d*w[:, None])


def gaussian_mask(shape, mu, C, theta=None):
    """exp(-0.5 (f-mu)^T C^-1 (f-mu)) on the (ft, fy, fx) grid (cycles/sample). If theta is given,
    the half-space opposite theta is set to 0 so the filter stays one-sided (Re even, Im odd)."""
    FT, FY, FX = freq_grid(shape)
    X = np.stack([FT, FY, FX], -1)/(2*np.pi)-mu
    G = np.exp(-0.5*np.einsum('...i,ij,...j->...', X, np.linalg.inv(C), X))
    if theta is not None:
        proj = np.cos(theta)*FX+np.sin(theta)*FY
        G = G*((proj > 0)+0.5*(proj == 0))
    return G


def half_space_leak(mu, C, theta):
    """Fraction of the (uncut) Gaussian's integral on the wrong side of the half-plane
    cos(theta) fx + sin(theta) fy < 0: the error of treating the filter as a full Gaussian."""
    from scipy.stats import norm
    n = np.array([0., np.sin(theta), np.cos(theta)])        # (ft, fy, fx) normal of the half-plane
    return norm.cdf(-(n@mu)/np.sqrt(n@C@n))


def make_fitted_bank(shape, thetas, sigmas, phis, sigma_t, f0=0.25, sigma_r=0.075, one_sided=True):
    """Fit a Cartesian 3D Gaussian to every product mask and return
    masks (K, S, T, H, W), mu (K, S, 3), C (K, S, 3, 3), R^2 of the fit (K, S), leak (K, S)."""
    P = make_masks(shape, thetas, sigmas, phis, sigma_t, f0=f0, sigma_r=sigma_r)
    K, S = P.shape[:2]
    masks = np.zeros_like(P)
    mu, C = np.zeros((K, S, 3)), np.zeros((K, S, 3, 3))
    r2, leak = np.zeros((K, S)), np.zeros((K, S))
    for k in range(K):
        for j in range(S):
            mu[k, j], C[k, j] = fit_gaussian(P[k, j], shape)
            masks[k, j] = gaussian_mask(shape, mu[k, j], C[k, j], thetas[k] if one_sided else None)
            G = masks[k, j]
            g = G*(P[k, j].ravel()@G.ravel())/(G.ravel()@G.ravel())
            r2[k, j] = 1-np.sum((P[k, j]-g)**2)/np.sum((P[k, j]-P[k, j].mean())**2)
            leak[k, j] = half_space_leak(mu[k, j], C[k, j], thetas[k])
    return masks, mu, C, r2, leak


# ---------------------------------------------------------------------------
# Separable Gaussian filters in spherical frequency coordinates (no fitting)
#   r = sqrt(fx^2 + fy^2 + ft^2),  theta = atan2(fy, fx),  phi = arccos(ft / r)
#   G(f) = exp(-(r - r0)^2 / 2 sigma_r^2) * exp(-d_theta^2 / 2 sigma_theta^2) * exp(-(phi - phi0)^2 / 2 sigma_phi^2)
# d_theta is cut at |d_theta| >= pi/2 so the filter is one-sided (Re even, Im odd).
# phi0 = 90 deg: static (ft = 0); phi0 < 90 deg: ft > 0 on the theta side.
# ---------------------------------------------------------------------------
def spherical_coords(shape):
    FT, FY, FX = freq_grid(shape)
    r = np.sqrt(FX**2+FY**2+FT**2)/(2*np.pi)                # cycles/sample
    theta = np.arctan2(FY, FX)
    phi = np.arccos(np.clip(np.divide(FT/(2*np.pi), r, out=np.zeros_like(r), where=r > 0), -1, 1))
    return r, theta, phi


def spherical_masks(shape, thetas, sigmas, phis0, sigma_phi, r0=0.25, sigma_r=0.075):
    """Masks (K, S, T, H, W). thetas: K azimuths (rad); sigmas: scalar, length-K or callable
    theta -> sigma_theta; phis0: S polar angles (rad, 0 = +ft axis, pi/2 = static);
    sigma_phi (rad); r0, sigma_r in cycles/sample."""
    r, th, ph = spherical_coords(shape)
    thetas = np.atleast_1d(thetas)
    sig = np.array([sigmas(t) for t in thetas]) if callable(sigmas) else np.broadcast_to(np.asarray(sigmas, float), thetas.shape)
    R = np.exp(-(r-r0)**2/(2*sigma_r**2))
    masks = np.zeros((len(thetas), len(phis0))+tuple(shape))
    for k, (t0, s) in enumerate(zip(thetas, sig)):
        d = _wrap(th-t0)
        A = np.exp(-d**2/(2*s**2))*(np.abs(d) < np.pi/2)
        for j, p0 in enumerate(phis0):
            masks[k, j] = R*A*np.exp(-(ph-p0)**2/(2*sigma_phi**2))
    masks[..., r == 0] = 0
    return masks
