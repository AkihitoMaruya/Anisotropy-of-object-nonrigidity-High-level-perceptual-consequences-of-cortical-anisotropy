#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template matching (Figure 5F/G: mean cosine similarity between the model flow directions and the template
directions) resolved by motion phase: the similarity to each template k is computed per frame, smoothed along phase
(Gaussian SD 3 frames), and the best k taken at each phase. Figure 5 flows (one band, Horn-Schunck alpha = 1 x
median speed), isotropic and anisotropic (width x2.5 cat) cortex, vertical (red) and horizontal (blue) rotation.
Right: vertical minus horizontal best k, with the phase of the largest difference within the Figure 5 fit window
(nearly edge-on phases excluded, where the direction fit is unreliable) marked.
Output: Images/Revision_R2/R2_86_fig5_k_by_phase.png. New analysis (cached flows).
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
from Revision_R2_heeger_gaussian_flow import FRAMES
from Revision_R2_two_rings_ME import Omega
D, P = TS.D, TS.P
plt.rcParams.update({'font.family': 'Arial', 'font.size': 22, 'axes.linewidth': 1.8})
COL = {'V': '#c0392b', 'H': '#2471a3'}
SMOOTH = 3
phase = np.degrees(Omega[FRAMES])


def k_by_phase(fl, o):
    u, v = D.unit(*D.flow_phys(fl))
    sim = np.array([np.nanmean(u*T[0]+v*T[1], 1) for T in D.TU[o]])     # (k, frames)
    sim = gaussian_filter1d(sim, SMOOTH, axis=1, mode='nearest')
    return D.KG[sim.argmax(0)]


if __name__ == '__main__':
    F = {o: np.load(P.A.data+f'heeger_fig5_hs_{o}.npz') for o in ('H', 'V')}
    K = {(o, kind): k_by_phase(F[o][f'{kind}_a1.0'], o) for o in ('V', 'H') for kind in ('A', 'U')}
    fig, ax = plt.subplots(1, 2, figsize=(22, 8))
    for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
        for o in ('V', 'H'):
            ax[0].plot(phase, K[(o, kind)], ls, color=COL[o], lw=3.5, alpha=.55 if kind == 'U' else 1, label=f'{"vertical" if o == "V" else "horizontal"} rotation, {cl}')
        ax[1].plot(phase, K[('V', kind)]-K[('H', kind)], ls, color='k', lw=3.5, label=cl)
    d = K[('V', 'A')]-K[('H', 'A')]; w = np.arange(len(phase))[P.FIT]
    top = w[d[P.FIT] >= d[P.FIT].max()-.01]; i = int(top[np.argmin(np.abs(top-top.mean()))])   # centre of the maximum (fit window)
    ax[1].axvline(phase[i], color='0.4', lw=1.5, ls='--', zorder=0)
    ax[1].annotate(f'largest difference\n{phase[top[0]]:.0f}–{phase[top[-1]]:.0f}° ({d[i]:.2f})', (phase[i], d[i]), xytext=(phase[i]+6, .05),
                   fontsize=17, va='bottom', ha='left')
    ax[0].set(ylabel='best k (template matching)', ylim=[-.05, 1.05], yticks=[0, .5, 1])
    ax[1].set(ylabel='vertical − horizontal best k', ylim=[-.05, .75])
    for a, t in zip(ax, ('Best k by motion phase', 'Nonrigidity anisotropy')):
        a.set(xlabel='Motion phase (deg)', xticks=[0, 45, 90, 135, 180], xlim=[0, 180]); a.set_title(t, loc='left', fontweight='bold')
        a.legend(fontsize=15, frameon=False, loc='lower center', bbox_to_anchor=(.5, 0) if a is ax[0] else (.7, .82))
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_fig5_k_by_phase.png', dpi=80)
    print(f'fit window {phase[P.FIT][0]:.0f}-{phase[P.FIT][-1]:.0f} deg; largest V-H difference {d[i]:.2f} at {phase[i]:.0f} deg; at 20 deg {d[np.argmin(np.abs(phase-20))]:.2f}; mean {d.mean():.2f}')
