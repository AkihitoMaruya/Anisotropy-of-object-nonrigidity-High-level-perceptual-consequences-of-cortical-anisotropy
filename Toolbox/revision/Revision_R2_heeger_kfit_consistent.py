#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotation (k = 0) vs wobble (k = 1) fit with CONSISTENT templates, and flows measured across the contour.
New analysis; it reads, but does not modify, the paper code and the earlier revision scripts.

Templates: for each k, the analytic image velocity of the ring (tilt +30 deg, with the cos(phi)
foreshortening, i.e. the same ring as the stimulus) is evaluated at the same contour locations as the
flows, and passed through exactly the same operator fields (Revision_R2_heeger_k_fit.fields: curl = -dP/dtheta,
div = dP/dtheta rotated +90 deg, def from the two shears), the same frames and the same normalisation
(shared maximum of |div|, |curl|, |def|). The motion with mix k is X_k(theta, Omega) = Ry(Omega) R_n(-k Omega) p(theta)
(Revision_R2_two_rings.ring_motion); the material point at a contour location shifts along the ring by k*Omega.

Flows: the Gaussian Heeger model (layout C, raw normalisation, D4G ring sigma 1.5 px, f0 0.12) is evaluated at
normal offsets r = -3..3 px from the ring's centre line (the D4G contour has width) and the flow vectors are
averaged over r at each contour point before computing the invariants; the centre line alone (r = 0) is kept
for comparison.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_heeger_k_fit as KF                               # fields, frames, paths (read only)
from Revision_R2_two_rings import ring_motion
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import TILT, sz_x, sz_y
from Revision_R2_D4G_ring import d4g_ring_video
from anisotropy_heeger import GaussianBank, cat_counts
from gaussian_angle_masks3D import cat_sigma

out, data = KF.out, KF.data
tilt = TILT['bottom']
Om = KF.Om                                                    # phases of FRAMES
tt = KF.tt                                                    # contour locations of the flows
OFFSETS = np.arange(-3, 4)                                    # px along the normal


# ---------------------------------------------------------------- invariants (same as KF, physical velocity)
def invariants_phys(U, V, orient):
    F = KF.fields(orient)
    I = {g: np.sum(f[0]*U+f[1]*V, axis=1) for g, f in F.items()}
    div, curl, dfm = I['div'][KF.KEEP], I['curl'][KF.KEEP], np.hypot(I['def1'], I['def2'])[KF.KEEP]
    n = max(np.abs(div).max(), np.abs(curl).max(), np.abs(dfm).max())
    return div/n, curl/n, dfm/n


def invariants_screen(flow, orient):
    return invariants_phys(-flow[..., 0], -flow[..., 1], orient)      # rendering flips both axes


# ---------------------------------------------------------------- consistent templates
def param_shift_sign():
    """R_n(-k Omega) p(theta) = p(theta + s k Omega): find s."""
    a = 0.3
    p_rot, _ = ring_motion(np.array([0.0]), 1.0, tilt, th=tt)             # at Omega = 0 no rotation
    P1, _ = ring_motion(np.array([a]), 1.0, tilt, th=tt)
    for s in (1, -1):
        P0, _ = ring_motion(np.array([a]), 0.0, tilt, th=tt+s*a)
        # compare with Ry(a) applied to the in-plane-rotated ring: positions must coincide as sets
        if np.allclose(np.sort(P1[0, 0]), np.sort(P0[0, 0]), atol=1e-6) and np.allclose(P1[0], ring_motion(np.array([a]), 1.0, tilt, th=tt)[0][0]):
            pass
    # direct test: material point theta under k=1 at Omega=a sits where the k=0 point theta - s*a sits
    for s in (1, -1):
        Pk, _ = ring_motion(np.array([a]), 1.0, tilt, th=tt)
        P0, _ = ring_motion(np.array([a]), 0.0, tilt, th=tt-s*a)
        if np.allclose(Pk, P0, atol=1e-6):
            return s
    raise RuntimeError('no parameter shift found')


S_SHIFT = param_shift_sign()


def analytic_velocity(k, orient):
    """Physical image velocity (d/dOmega) of motion mix k at the contour locations tt, frames FRAMES."""
    U = np.zeros((len(Om), len(tt)))
    V = np.zeros_like(U)
    for i, o in enumerate(Om):
        th_material = tt+S_SHIFT*k*o                          # material points now at locations tt
        _, vel = ring_motion(np.array([o]), k, tilt, th=th_material)
        U[i], V[i] = vel[0, 0], vel[0, 1]
    if orient == 'V':
        U, V = -V, U                                           # rotate by +90 deg, as stimulus_and_fields
    return U, V


KG = np.round(np.arange(0, 1.2001, 0.01), 2)
TPL = {o: [invariants_phys(*analytic_velocity(k, o), o) for k in KG] for o in ('H', 'V')}


def best_k(inv, orient):
    cost = np.array([sum(np.sum((a-b)**2) for a, b in zip(inv, T)) for T in TPL[orient]])
    i = cost.argmin()
    lo, hi = KG[max(i-1, 0)], KG[min(i+1, len(KG)-1)]
    f = lambda k: sum(np.sum((a-b)**2) for a, b in zip(inv, invariants_phys(*analytic_velocity(k, orient), orient)))
    r = minimize_scalar(f, bounds=(lo, hi), method='bounded') if hi > lo else None
    return (r.x, r.fun) if r is not None else (KG[i], cost[i])


# ---------------------------------------------------------------- flows across the contour
def offset_points(L, n_scr, r):
    rows = L[..., 0]-r*n_scr[..., 1]                           # screen y up = -row
    cols = L[..., 1]+r*n_scr[..., 0]
    return np.clip(np.round(rows), 0, sz_y-1).astype(int), np.clip(np.round(cols), 0, sz_x-1).astype(int)


K16 = 16
thetas = np.arange(K16)*2*np.pi/K16
S_MEAN = cat_sigma(thetas).mean()
LAYOUT_C = dict(phis=np.radians([-60, -30, 0, 30, 60]), f0=0.12, sigma_r=0.04, sigma_t=np.radians(15))


def bank(kind):
    kw = dict(f0=LAYOUT_C['f0'], sigma_r=LAYOUT_C['sigma_r'], sigma_t=LAYOUT_C['sigma_t'])
    if kind == 'iso':
        return GaussianBank(thetas, np.full(K16, S_MEAN), LAYOUT_C['phis'], **kw)
    beta = {'aniso1': 1.0, 'aniso4': 4.0}[kind]
    w = S_MEAN*(1+beta*(cat_sigma(thetas)/S_MEAN-1))
    return GaussianBank(thetas, w, LAYOUT_C['phis'], weights=cat_counts(thetas), normalize='raw', **kw)


def flows_across(orient, kinds=('iso', 'aniso1', 'aniso4')):
    cache = data+f'heeger_layoutC_across_{orient}.npz'
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in kinds}
    video = d4g_ring_video(orient, sigma=1.5)
    Vf = np.fft.fftshift(np.fft.fftn(video))
    L, _, n_scr = ring_points(orient)
    L, n_scr = L[FRAMES], n_scr[FRAMES]
    rr, cc = zip(*[offset_points(L, n_scr, r) for r in OFFSETS])
    pts = (np.tile(np.repeat(FRAMES, NP), len(OFFSETS)), np.concatenate([x.ravel() for x in rr]),
           np.concatenate([x.ravel() for x in cc]))
    res = {}
    for kind in kinds:
        t0 = time.time()
        b = bank(kind)
        res[kind] = b.flow(b.energies(video, pts, Vf)).reshape(len(OFFSETS), len(FRAMES), NP, 2)
        print(f'  {orient} {kind}: {(time.time()-t0)/60:.1f} min', flush=True)
    np.savez(cache, **res)
    return res


if __name__ == '__main__':
    print(f'template parameter shift sign: {S_SHIFT:+d}')
    rows, table = {}, []
    for o in ('H', 'V'):
        U, V = analytic_velocity(0.0, o)
        L, v_true, n_scr = ring_points(o)
        vt, n = v_true[FRAMES], n_scr[FRAMES]
        vn = np.sum(vt*n, -1, keepdims=True)*n
        rows[('check: true rigid rotation (analytic)', o)] = invariants_phys(U, V, o)
        rows[('check: true velocity from the rendered ring', o)] = invariants_screen(vt, o)
        rows[('normal flow', o)] = invariants_screen(vn, o)
        F = flows_across(o)
        c = list(OFFSETS).index(0)
        for kind, lab in (('iso', 'isotropic'), ('aniso1', 'anisotropic ×1'), ('aniso4', 'anisotropic ×4')):
            rows[(f'{lab}, centre line', o)] = invariants_screen(F[kind][c], o)
            rows[(f'{lab}, averaged across r', o)] = invariants_screen(F[kind].mean(0), o)
        P = [np.load(current_folder+f'/Toolbox/Data/{o}_{g}_aniso_im_stretch_1.0.npy')[KF.KEEP] for g in ('div', 'curl', 'def')]
        nP = max(np.abs(p).max() for p in P)
        rows[("paper's saved Fig 6 invariants (aniso)", o)] = tuple(p/nP for p in P)

    names = list(dict.fromkeys(k[0] for k in rows))
    fits = {}
    for nm in names:
        for o in ('H', 'V'):
            fits[(nm, o)] = best_k(rows[(nm, o)], o)
        print(f'{nm:45s}: k  H {fits[(nm, "H")][0]:.2f}  V {fits[(nm, "V")][0]:.2f}   '
              f'(residual H {fits[(nm, "H")][1]:.2f}, V {fits[(nm, "V")][1]:.2f})')
    np.save(data+'heeger_kfit_consistent.npy', dict(fits=fits), allow_pickle=True)

    show = ['normal flow', 'isotropic, averaged across r', 'anisotropic ×1, averaged across r',
            'anisotropic ×4, averaged across r', "paper's saved Fig 6 invariants (aniso)"]
    cols = (('r', 'Divergence'), ('g', 'Curl'), ('b', 'Deformation'))
    phase = np.degrees(KF.Om_fit)
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(len(show), 2, figsize=(17, 4.4*len(show)), sharex=True, sharey=True)
    for r, nm in enumerate(show):
        for c, o in enumerate(('H', 'V')):
            a = ax[r, c]
            k, res = fits[(nm, o)]
            tpl = invariants_phys(*analytic_velocity(k, o), o)
            for (col, lab), y, t in zip(cols, rows[(nm, o)], tpl):
                a.plot(phase, y, 'o', color=col, ms=3.5, alpha=.6, label=f'{lab} (flow)')
                a.plot(phase, t, '--', color=col, lw=3, label=f'{lab} (template)')
            a.axhline(0, color='gray', lw=.8)
            a.set_title(f'{nm} — {"Horizontal" if o == "H" else "Vertical"}: k = {k:.2f} (residual {res:.1f})', fontsize=12)
            a.set_ylim([-1.15, 1.15]); a.set_xticks([0, 45, 90, 135, 180])
            if c == 0:
                a.set_ylabel('Normalised invariant')
            if r == len(show)-1:
                a.set_xlabel('Motion phase (deg)')
    ax[0, 0].legend(fontsize=9, ncol=3, loc='lower center')
    fig.suptitle('Consistent templates (same fields, points, frames and normalisation as the flows); k = 0 rotation, 1 wobble', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, .985])
    fig.savefig(out+'R2_26_heeger_kfit_consistent.png', dpi=85)
