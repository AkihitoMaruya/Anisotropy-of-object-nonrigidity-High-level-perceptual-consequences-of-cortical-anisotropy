#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corrected SP3D optic-flow / global-motion computation.

This replaces the analytic closed form used in `texture_synthesis3D_flow.py`
(`_tildeJ_K4` / `_predicted_R`), which assumed Gaussian-derivative energy filters
— NOT the steerable pyramid's actual log-raised-cosine radial × cos^(K-1) angular
filters. That mismatch makes the predicted response R(v) wrong and the per-voxel
velocity fit rail at the grid edge.

Two corrected pieces:

1) expected_R_analytic(v) — the EXPECTED motion energy for moving white noise,
   evaluated as the defining integral over the pyramid's OWN filters (no closed
   form, no Monte-Carlo):
     - white noise + rigid translation at v puts all stimulus energy on the
       motion plane  omega_t = -v . omega_spatial, which cuts the frequency
       sphere in a GREAT CIRCLE;
     - the separable radial part integrates to a v-independent constant (absorbed
       by the per-orientation amplitude), so
         R(v;b,s)  ∝  ∮_{great circle}  A_b(theta_xy)^2 A_s(theta_xz)^2  dphi
       with A_b, A_s the pyramid's two-sided cos^(K-1)/cos^(S-1) angular filters.
   Because the pyramid downsamples space and time together, v is in px/frame and
   R is scale-independent. Validated: recovers a clean grating's velocity (the
   closed form railed).

2) Global motion = LOCAL-LIKELIHOOD INFORMATION-MATRIX combination + slow-motion
   prior (faithful Weiss/Heeger). Each voxel's normalized local likelihood
   exp(-chi(v)/temp) is moment-matched to a Gaussian (mean mu, covariance Sigma);
   the information Lambda = Sigma^-1. An ambiguous 1-D-grating voxel has a
   likelihood RIDGE -> huge variance along the ridge -> ~0 precision
   perpendicular, so it cannot bias that direction; the aperture terminators are
   2-D -> full precision -> they set the perpendicular direction. Combine
   Lambda = sum_i Lambda_i (+ prior), v = (sum Lambda)^-1 sum Lambda_i mu_i.
   Scales combine by summing their information matrices (a no-energy scale gets a
   tiny Lambda and is auto-down-weighted).

@author: ported/derived with Aki, 2026-06-22
"""
import numpy as np
import torch

from steerable_pyramid3D import Steerable_Pyramid_Freq3D


# ---------- expected motion energy R(v) : great-circle integral ----------------
def _ang_filter(k, K, ang):
    """Pyramid two-sided cos^(K-1) angular filter (unit amplitude)."""
    def lobe(center):
        d = (ang - center + np.pi) % (2 * np.pi) - np.pi
        out = np.zeros_like(ang)
        m = np.abs(d) < np.pi / 2
        out[m] = np.cos(d[m]) ** (K - 1)
        return out
    return lobe(k * np.pi / K) + lobe(np.pi * (k - K) / K)


_R_CACHE = {}


def expected_R_analytic(v_grid, K=4, S=4, n_phi=512):
    """Expected motion-energy template R(v;b,s) on the (V, K, S) grid, via the
    great-circle integral over the motion plane.  v in px/frame, scale-independent."""
    key = (tuple(np.round(v_grid, 6)), K, S, n_phi)
    if key in _R_CACHE:
        return _R_CACHE[key]
    VX, VY = np.meshgrid(v_grid, v_grid, indexing="xy")
    fx, fy = VX.ravel(), VY.ravel()
    R = np.zeros((fx.size, K, S))
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    cph, sph = np.cos(phi), np.sin(phi)
    for i in range(fx.size):
        n = np.array([fx[i], fy[i], 1.0]); n /= np.linalg.norm(n)
        a = np.array([1., 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1., 0])
        e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        FX = cph * e1[0] + sph * e2[0]
        FY = cph * e1[1] + sph * e2[1]
        FZ = cph * e1[2] + sph * e2[2]
        txy = np.arctan2(FY, FX)
        for b in range(K):
            cb, sb = np.cos(b * np.pi / K), np.sin(b * np.pi / K)
            txz = np.arctan2(FZ, cb * FX + sb * FY)
            Ab2 = _ang_filter(b, K, txy) ** 2
            for t in range(S):
                R[i, b, t] = float(np.sum(Ab2 * _ang_filter(t, S, txz) ** 2))
    _R_CACHE[key] = (R, VX, VY)
    return R, VX, VY


# ---------- motion energy from the pyramid -------------------------------------
_PYR_CACHE = {}


def motion_energy_all_scales(frames, K=4, S=4, height=3, downsample=True):
    """One pyramid pass on a (T,H,W) video -> {scale: (M2, K, S)} per oriented scale."""
    vid = torch.as_tensor(np.asarray(frames, dtype=np.float32))
    T, H, W = vid.shape
    key = ((T, H, W), K, S, height, downsample)
    if key not in _PYR_CACHE:
        _PYR_CACHE[key] = Steerable_Pyramid_Freq3D(
            (T, H, W), height=height, order=K - 1, order_temp=S - 1,
            is_complex=True, downsample=downsample,
            hilbert_axes=("spatial", "temporal"))
    pyr = _PYR_CACHE[key]
    out = {}
    with torch.no_grad():
        coeffs = pyr.forward(vid.unsqueeze(0).unsqueeze(0))
        Kb, St = pyr.num_orientations, pyr.num_orientations_temp
        for s in range(pyr.num_scales):
            me = [pyr.motion_energy_sq(coeffs, s, b, t).squeeze()
                  for b in range(Kb) for t in range(St)]
            M2 = torch.stack(me, dim=-1).reshape(*me[0].shape, Kb, St)
            out[s] = (M2.cpu().numpy().astype(np.float64), Kb, St)
    return out


# ---------- global motion : information-matrix combination + slow prior ---------
def info_components(M2, Kb, St, R, VX, VY, *, energy_frac=0.6, temp_frac=0.15, chunk=256):
    """Accumulated info SL = sum Lambda_i and SLm = sum Lambda_i mu_i over the
    energetic voxels (so scales can be combined by summing these)."""
    fx, fy = VX.ravel(), VY.ravel()
    M2f = M2.reshape(-1, Kb, St)
    E = M2f.sum(axis=(1, 2))
    M2k = M2f[E >= np.quantile(E, energy_frac)]
    mbar = M2k.sum(-1); msq = (M2k ** 2).sum(-1)
    Rbar = R.sum(-1) + 1e-12; Rsq = (R ** 2).sum(-1)
    SL = np.zeros((2, 2)); SLm = np.zeros(2)
    for c0 in range(0, len(M2k), chunk):
        ch = slice(c0, c0 + chunk)
        MR = np.einsum("nbs,vbs->nvb", M2k[ch], R)
        al = mbar[ch][:, None, :] / Rbar[None]
        chi = (msq[ch][:, None, :] - 2 * al * MR + al ** 2 * Rsq[None]).sum(-1)
        chi = chi - chi.min(1, keepdims=True)
        temp = np.maximum(1e-9, temp_frac * np.ptp(chi, 1, keepdims=True))
        w = np.exp(-chi / temp); w /= w.sum(1, keepdims=True)
        mux = (w * fx).sum(1); muy = (w * fy).sum(1)
        dx = fx[None] - mux[:, None]; dy = fy[None] - muy[:, None]
        cxx = (w * dx * dx).sum(1) + 1e-3
        cyy = (w * dy * dy).sum(1) + 1e-3
        cxy = (w * dx * dy).sum(1)
        det = cxx * cyy - cxy ** 2
        Lxx, Lyy, Lxy = cyy / det, cxx / det, -cxy / det
        SL[0, 0] += Lxx.sum(); SL[1, 1] += Lyy.sum(); SL[0, 1] += Lxy.sum()
        SLm[0] += (Lxx * mux + Lxy * muy).sum()
        SLm[1] += (Lxy * mux + Lyy * muy).sum()
    SL[1, 0] = SL[0, 1]
    return SL, SLm


def solve_info(SL, SLm, prior_rel=0.0):
    """v = (SL + lam I)^-1 SLm,  lam = prior_rel * trace(SL)/2  (slow-motion prior)."""
    lam = prior_rel * np.trace(SL) / 2.0
    v = np.linalg.solve(SL + lam * np.eye(2), SLm)
    return float(v[0]), float(v[1])


def global_velocity(frames, *, scales=(0, 1, 2), v_max=3.0, n_v=41, prior_rel=0.0,
                    energy_frac=0.6, temp_frac=0.15, K=4, S=4, height=3):
    """End-to-end faithful global velocity (vx, vy) in px/frame for a (T,H,W) video.
    `scales` are combined by summing their information matrices."""
    R, VX, VY = expected_R_analytic(np.linspace(-v_max, v_max, n_v), K, S)
    allM = motion_energy_all_scales(frames, K=K, S=S, height=height)
    SL = np.zeros((2, 2)); SLm = np.zeros(2)
    for s in scales:
        M2, Kb, St = allM[s]
        sl, slm = info_components(M2, Kb, St, R, VX, VY,
                                  energy_frac=energy_frac, temp_frac=temp_frac)
        SL += sl; SLm += slm
    return solve_info(SL, SLm, prior_rel)
