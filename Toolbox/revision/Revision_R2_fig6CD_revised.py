#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised Figure 6C (isotropic cortex) and 6D (anisotropic cortex, width x2.5 cat): deformation over curl,
atan2(def, curl), of the model optic flows (dots; Figure 5 flows: one band, Horn-Schunck smoothing alpha = 1 x
median speed, Revision_R2_fig5_hs_smooth.py) and of the best-fitting rotation-to-wobble template (lines), for
horizontal and vertical rotation; shaded: deviation between the model flow and the
best fit (SMOOTH > 0 smooths the model flow along phase for display only). Invariants as in the revised Figure 6B (Revision_R2_fig6B_revised.py): contour
integrals of the operator fields along the ring; the area normalisation cancels in the ratio. k is fitted by least
squares on the angle over all phases. E: vertical minus horizontal best-fit
curve for each cortex, with the phase of the largest difference marked. New analysis (cached flows).
Output: Images/Revision_R2/R2_86_fig6CD.png
"""
import os, sys
os.environ.setdefault('K_FIT_TRIM', '10')
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_heeger_temperature_sweep as TS
import Revision_R2_fig6B_revised as B
C, A = B.C, B.A
P = TS.P
plt.rcParams.update({'font.family': 'Arial', 'font.size': 22, 'axes.linewidth': 1.8,
                     'xtick.major.width': 1.8, 'ytick.major.width': 1.8, 'xtick.major.size': 7, 'ytick.major.size': 7})
KG = np.round(np.arange(0, 1.5001, 0.01), 2)
FIT = slice(None)                                                 # all phases
SMOOTH = 0                                                        # display only: Gaussian SD (frames) along phase; 0 = none
COL = {'V': '#c0392b', 'H': '#2471a3'}                            # vertical red, horizontal blue


def ratio(div, curl, dfm):
    return np.degrees(np.arctan2(dfm, curl))


def flow_ratio(fl, o):
    U, V = -fl[..., 0], -fl[..., 1]                                    # screen -> ring coordinates
    I = {g: np.sum(f[0]*U+f[1]*V, 1) for g, f in C.KF.fields(o).items()}
    return ratio(I['div'], I['curl'], np.hypot(I['def1'], I['def2']))


TPL = {o: np.array([ratio(*B.per_area_templates(k, o)) for k in KG]) for o in ('H', 'V')}


def best_k(r, o):
    cost = np.nansum((TPL[o][:, FIT]-r[None, FIT])**2, 1)
    return KG[cost.argmin()]


if __name__ == '__main__':
    F = {o: np.load(P.A.data+f'heeger_fig5_hs_{o}.npz') for o in ('H', 'V')}
    fig, ax = plt.subplots(1, 3, figsize=(33, 8))
    FITC, EMP = {}, {}
    for a, kind, title in ((ax[0], 'U', 'C   Isotropic cortex'), (ax[1], 'A', 'D   Anisotropic cortex')):
        for o in ('V', 'H'):
            r = flow_ratio(F[o][f'{kind}_a1.0'], o); k = best_k(r, o)
            lab = 'horizontal' if o == 'H' else 'vertical'
            FITC[(kind, o)] = TPL[o][list(KG).index(k)]; EMP[(kind, o)] = r = gaussian_filter1d(r, SMOOTH, mode='nearest') if SMOOTH else r
            a.fill_between(B.phase, r, FITC[(kind, o)], color=COL[o], alpha=.25 if kind == 'A' else .18, lw=0)
            a.plot(B.phase, FITC[(kind, o)], '-', color=COL[o], lw=3.5, alpha=1 if kind == 'A' else .55, label=f'{lab} rotation, best fit k = {k:.2f}')
            print(f'{kind} {o}: best k {k:.2f}')
        a.axhline(45, color='0.5', lw=1.2, ls=':', zorder=0); a.axhline(90, color='0.5', lw=1.2, ls='--', zorder=0)
        a.set(xlabel='Motion phase (deg)', xticks=[0, 45, 90, 135, 180], xlim=[0, 180], ylim=[0, 180], yticks=[0, 45, 90, 135, 180])
        a.set_title(title, loc='left', fontweight='bold')
        a.legend(fontsize=16, frameon=False, loc='upper left')
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel('Def / Curl (deg)')
    a = ax[2]
    for kind, ls, lab in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
        d = FITC[(kind, 'V')]-FITC[(kind, 'H')]
        a.fill_between(B.phase, EMP[(kind, 'V')]-EMP[(kind, 'H')], d, color='k', alpha=.15, lw=0)
        a.plot(B.phase, d, ls, color='k', lw=3.5, label=lab)
    d = FITC[('A', 'V')]-FITC[('A', 'H')]; i = int(np.argmax(d))
    a.axvline(B.phase[i], color='0.4', lw=1.5, ls='--', zorder=0)
    a.annotate(f'largest difference\nat {B.phase[i]:.0f}° ({d[i]:.0f}°)', (B.phase[i], d[i]), xytext=(B.phase[i]+6, 10),
               fontsize=17, va='bottom', ha='left')
    a.set(xlabel='Motion phase (deg)', xticks=[0, 45, 90, 135, 180], xlim=[0, 180], ylim=[-5, 75],
          ylabel='vertical − horizontal best fit (deg)')
    a.set_title('E   Nonrigidity anisotropy', loc='left', fontweight='bold')
    a.legend(fontsize=16, frameon=False, loc='upper left')
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    print('E: max difference', round(d[i], 1), 'deg at phase', round(B.phase[i], 1), '; at 20 deg', round(d[np.argmin(np.abs(B.phase-20))], 1))
    ax[0].text(178, 47, 'def = curl', ha='right', va='bottom', fontsize=16, color='0.4')
    ax[0].text(178, 92, 'curl = 0', ha='right', va='bottom', fontsize=16, color='0.4')
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_fig6CD.png', dpi=85)
