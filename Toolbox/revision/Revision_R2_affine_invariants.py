#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard (un-normalised) first-order invariants of the flow on the rotating ring. For each frame an affine
flow v = A x + b is fitted by least squares to all contour points, and
    div = du/dx + dv/dy,  curl = dv/dx - du/dy,  def = sqrt((du/dx - dv/dy)^2 + (du/dy + dv/dx)^2)   [1/frame]
Screen coordinates (x right, y up, px); the 180 deg flip between screen and ring coordinates leaves A unchanged.
Templates: the analytic rotation/wobble velocity (mix k) at the same contour points (exactly affine for a rigid
planar ring under orthographic projection), converted to px/frame. Flows: normal flow, and the Gaussian Heeger
flows (layout C, raw normalisation, D4G ring sigma 1.5 px), averaged across the contour (r = -3..3 px).
New analysis; reads existing code and data only.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_heeger_kfit_consistent as C
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES
from Revision_R2_two_rings import ring_motion
from Revision_R2_two_rings_ME import TILT, Omega

out, data = C.out, C.data
dO = Omega[1]-Omega[0]                                       # rad of rotation per frame
PX = C.sz_x/2.4*(C.sz_x-1)/C.sz_x                            # px per ring unit (make_rotating_stim mapping)
KEEP = C.KF.KEEP


def affine(Xs, Ys, U, V):
    """Per frame least-squares affine fit -> div, curl, def (1/frame)."""
    div, curl, dfm = [], [], []
    for x, y, u, v in zip(Xs, Ys, U, V):
        M = np.stack([np.ones_like(x), x, y], 1)
        a = np.linalg.lstsq(M, u, rcond=None)[0]
        b = np.linalg.lstsq(M, v, rcond=None)[0]
        ux, uy, vx, vy = a[1], a[2], b[1], b[2]
        div.append(ux+vy); curl.append(vx-uy); dfm.append(np.hypot(ux-vy, uy+vx))
    return np.array(div)[KEEP], np.array(curl)[KEEP], np.array(dfm)[KEEP]


def screen_positions(o):
    L, v_true, n = ring_points(o)
    return L[FRAMES, :, 1], -L[FRAMES, :, 0], v_true[FRAMES], n[FRAMES]


def template(k, o):
    """Analytic velocity of mix k at the contour points, px/frame, in ring coordinates; positions likewise."""
    U, V = C.analytic_velocity(k, o)                         # d/dOmega, ring units
    pos, _ = ring_motion(C.Om, 0.0, TILT['bottom'], th=C.tt)
    x, y = pos[:, 0], pos[:, 1]
    if o == 'V':
        x, y = -y, x
    return affine(x*PX, y*PX, U*PX*dO, V*PX*dO)


if __name__ == '__main__':
    F = {o: np.load(data+f'heeger_layoutC_across_{o}.npz') for o in ('H', 'V')}
    res = {}
    for o in ('H', 'V'):
        X, Y, vt, n = screen_positions(o)
        vn = np.sum(vt*n, -1, keepdims=True)*n
        res[(o, 'true velocity (rendered ring)')] = affine(X, Y, vt[..., 0], vt[..., 1])
        res[(o, 'normal flow')] = affine(X, Y, vn[..., 0], vn[..., 1])
        for kind, lab in (('iso', 'Heeger, isotropic'), ('aniso1', 'Heeger, anisotropic ×1'), ('aniso4', 'Heeger, anisotropic ×4')):
            fl = F[o][kind].mean(0)
            res[(o, lab)] = affine(X, Y, fl[..., 0], fl[..., 1])
        for k in (0.0, 0.5, 1.0):
            res[(o, f'template k = {k:g}')] = template(k, o)

    phase = np.degrees(C.Om[KEEP])
    style = {'template k = 0': ('k', '-', 3), 'template k = 0.5': ('0.5', '-', 3), 'template k = 1': ('k', '--', 3),
             'true velocity (rendered ring)': ('tab:green', ':', 2.5), 'normal flow': ('0.35', 'o', 0),
             'Heeger, isotropic': ('tab:blue', 'o', 0), 'Heeger, anisotropic ×1': ('tab:orange', 'o', 0),
             'Heeger, anisotropic ×4': ('tab:red', 'o', 0)}
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(2, 3, figsize=(21, 11), sharex=True)
    for r, o in enumerate(('H', 'V')):
        for c, (g, lab) in enumerate((('div', 'Divergence'), ('curl', 'Curl'), ('def', 'Deformation'))):
            a = ax[r, c]
            for name, (col, ls, lw) in style.items():
                y = res[(o, name)][c]*1e3
                if ls == 'o':
                    a.plot(phase, y, 'o', color=col, ms=3.5, alpha=.7, label=name)
                else:
                    a.plot(phase, y, ls, color=col, lw=lw, label=name)
            a.axhline(0, color='gray', lw=.8)
            a.set(title=f'{"Horizontal" if o == "H" else "Vertical"} rotation — {lab}', xticks=[0, 45, 90, 135, 180])
            if c == 0:
                a.set_ylabel('×10⁻³ per frame')
            if r == 1:
                a.set_xlabel('Motion phase (deg)')
    ax[0, 0].legend(fontsize=10)
    fig.suptitle('Standard first-order invariants from an affine fit to the flow on the ring (no normalisation)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(out+'R2_28_affine_invariants.png', dpi=85)
    for o in ('H', 'V'):
        for name in style:
            d, cu, df = res[(o, name)]
            print(f'{o} {name:32s}: mean div {d.mean()*1e3:+6.2f}  curl {cu.mean()*1e3:+6.2f}  def {df.mean()*1e3:6.2f}  (×10⁻³/frame)')
