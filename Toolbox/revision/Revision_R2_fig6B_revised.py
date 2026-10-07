#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised Figure 6B: rotation-to-wobble templates of divergence, curl and deformation as functions of motion phase,
normalised by the contour: the paper's contour integrals (operator fields along the ring, Figure6.py) divided by
the ring's enclosed area, which by Green's theorem gives the standard invariants (div = du/dx + dv/dy, etc.),
instead of dividing by the maximum over phases. k = 0 (rotation) to 1 (wobble) in steps of 0.05, horizontal
rotation, per frame (1.4 deg of rotation per frame), at the model's frames (phases 7-172 deg; the enclosed area -> 0
as the ring turns edge-on at 0 and 180 deg). Fourth panel: deformation over curl as the angle atan2(def, curl)
(45 deg: def = curl, 90 deg: curl = 0, > 90 deg: negative curl; the ratio itself diverges where curl crosses 0);
the area cancels, so it is the same with any normalisation. New analysis; reads existing code only.
Output: Images/Revision_R2/R2_86_fig6B.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_affine_invariants as A
from Revision_R2_two_rings import ring_motion
from Revision_R2_two_rings_ME import TILT
C = A.C
plt.rcParams.update({'font.family': 'Arial', 'font.size': 22, 'axes.linewidth': 1.8,
                     'xtick.major.width': 1.8, 'ytick.major.width': 1.8, 'xtick.major.size': 7, 'ytick.major.size': 7})

pos, _ = ring_motion(C.Om, 0.0, TILT['bottom'], th=C.tt)
x, y = pos[:, 0], pos[:, 1]
area = 0.5*np.abs(np.sum(x*np.roll(y, -1, 1)-np.roll(x, -1, 1)*y, 1))   # enclosed area (shoelace), per phase
dth = 2*np.pi/len(C.tt)
phase = np.degrees(C.Om)
KS = np.round(np.arange(0, 1.001, 0.05), 2)


def per_area_templates(k, o='H'):
    """div, curl, def of motion mix k: contour integral / enclosed area, per frame."""
    U, V = C.analytic_velocity(k, o)
    I = {g: np.sum(f[0]*U+f[1]*V, 1)*dth for g, f in C.KF.fields(o).items()}
    return np.array([I['div'], I['curl'], np.hypot(I['def1'], I['def2'])])/area*A.dO


if __name__ == '__main__':
    cols = cm.jet(np.linspace(0, 1, len(KS)))
    fig, ax = plt.subplots(1, 4, figsize=(32, 7.2))
    for k, col in zip(KS, cols):
        T = per_area_templates(k)
        for a, v in zip(ax, T):
            a.plot(phase, v, color=col, lw=3)
        ax[3].plot(phase, np.degrees(np.arctan2(T[2], T[1])), color=col, lw=3)
    for a, lab in zip(ax, ('Div (per frame)', 'Curl (per frame)', 'Def (per frame)', 'Def / Curl (deg)')):
        a.axhline(0, color='0.5', lw=1.2, zorder=0) if not lab.startswith('Def /') else None
        a.set(ylabel=lab, xlabel='Motion phase (deg)', xticks=[0, 45, 90, 135, 180], xlim=[0, 180])
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    ax[3].axhline(45, color='0.5', lw=1.2, ls=':', zorder=0); ax[3].axhline(90, color='0.5', lw=1.2, ls='--', zorder=0)
    ax[3].set(ylim=[0, 180], yticks=[0, 45, 90, 135, 180])
    ax[3].text(3, 41, 'def = curl', ha='left', va='top', fontsize=17, color='0.4')
    ax[3].text(3, 92, 'curl = 0', ha='left', va='bottom', fontsize=17, color='0.4', bbox=dict(fc='white', ec='none', pad=1), zorder=5)
    fig.suptitle('Gradients of rotation to wobbling', fontsize=26, y=1.03)
    fig.subplots_adjust(wspace=.32)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cm.jet), ax=ax, fraction=.018, pad=.012)
    cb.set_label('k'); cb.set_ticks([0, .5, 1]); cb.set_ticklabels(['0 (rotation)', '0.5', '1 (wobble)'])
    fig.savefig(A.out+'R2_86_fig6B.png', dpi=90, bbox_inches='tight')
    k0, k1 = per_area_templates(0.0), per_area_templates(1.0)
    print('max |div(k=1) - div(k=0)| =', np.abs(k1[0]-k0[0]).max())
