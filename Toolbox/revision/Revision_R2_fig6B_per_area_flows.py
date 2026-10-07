#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-area invariants (contour integral / enclosed area = standard div, curl, def; no normalisation) of the model
optic flows on the rotating D4G ring, with the best-fitting rotation/wobble template (least squares on the raw
curves, first/last 4 frames excluded from the fit). Flows: Gaussian Heeger, layout C, raw normalisation, averaged
across the contour (r = -3..3 px): isotropic, anisotropic x1 and x4. New analysis; reads existing code and data.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_affine_invariants as A
from Revision_R2_two_rings import ring_motion
from Revision_R2_two_rings_ME import TILT
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES
C = A.C

pos, _ = ring_motion(C.Om, 0.0, TILT['bottom'], th=C.tt)
x, y = pos[:, 0], pos[:, 1]
area = 0.5*np.abs(np.sum(x*np.roll(y, -1, 1)-np.roll(x, -1, 1)*y, 1))
dth = 2*np.pi/len(C.tt)
phase = np.degrees(C.Om)
FIT = C.KF.KEEP                                                  # frames used for fitting


def per_area(U, V, o):
    """U, V: velocity in ring units per frame (ring coordinates). Returns div, curl, def per frame."""
    F = C.KF.fields(o)
    I = {g: np.sum(f[0]*U+f[1]*V, 1)*dth for g, f in F.items()}
    return np.array([I['div']/area, I['curl']/area, np.hypot(I['def1'], I['def2'])/area])


def template(k, o):
    U, V = C.analytic_velocity(k, o)                             # per radian of rotation
    return per_area(U*A.dO, V*A.dO, o)


def flow_inv(fl, o):
    return per_area(-fl[..., 0]/A.PX, -fl[..., 1]/A.PX, o)       # screen px/frame -> ring units/frame


KG = np.round(np.arange(0, 1.5001, 0.01), 2)
TPL = {o: np.array([template(k, o) for k in KG]) for o in ('H', 'V')}


def best_k(inv, o, free_scale=False):
    y = inv[:, FIT].ravel()
    T = TPL[o][:, :, FIT].reshape(len(KG), -1)
    if free_scale:
        s = (T@y)/np.sum(T*T, 1)
        cost = np.sum((y[None]/s[:, None]-T)**2, 1)
    else:
        s = np.ones(len(KG)); cost = np.sum((y[None]-T)**2, 1)
    i = cost.argmin()
    return KG[i], 1/s[i]


if __name__ == '__main__':
    runs = (('iso', 'isotropic', 'tab:blue'), ('aniso1', 'anisotropic ×1', 'tab:orange'), ('aniso4', 'anisotropic ×4', 'tab:red'))
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(2, 3, figsize=(22, 12), sharex=True)
    for r, o in enumerate(('H', 'V')):
        Fl = np.load(A.data+f'heeger_layoutC_across_{o}.npz')
        L, vt, n = ring_points(o)
        vn = np.sum(vt[FRAMES]*n[FRAMES], -1, keepdims=True)*n[FRAMES]
        inv_n = flow_inv(vn, o); kn, _ = best_k(inv_n, o)
        print(f'{o} normal flow: k = {kn:.2f}')
        for kind, lab, col in runs:
            inv = flow_inv(Fl[kind].mean(0), o)
            k, _ = best_k(inv, o)
            kf, sf = best_k(inv, o, True)
            print(f'{o} {lab:15s}: best k = {k:.2f} (raw);  {kf:.2f} with free scale (flow ×{sf:.2f})')
            T = template(k, o)
            for c in range(3):
                ax[r, c].plot(phase, inv[c]*1e3, 'o', color=col, ms=3.5, alpha=.6)
                ax[r, c].plot(phase, T[c]*1e3, '-', color=col, lw=2.5, label=f'{lab}: best k = {k:.2f}')
        for c, lab in enumerate(('Divergence', 'Curl', 'Deformation')):
            ax[r, c].axhline(0, color='gray', lw=.8)
            ax[r, c].axvspan(phase[0], phase[FIT][0], color='0.9'); ax[r, c].axvspan(phase[FIT][-1], phase[-1], color='0.9')
            ax[r, c].set(title=f'{"Horizontal" if o == "H" else "Vertical"} rotation — {lab}', xticks=[0, 45, 90, 135, 180])
        ax[r, 0].set_ylabel('×10⁻³ per frame')
        ax[r, 1].legend(fontsize=11)
    for c in range(3):
        ax[1, c].set_xlabel('Motion phase (deg)')
    fig.suptitle('Model optic flows (dots) and best-fitting rotation/wobble template (lines): contour integral ÷ area, no normalisation'
                 '\n(grey bands: frames excluded from the fit)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(A.out+'R2_34_fig6B_per_area_flows.png', dpi=85)
