#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary: how much energy each filter collects as a function of motion direction, and the population sum.
A grating at the filters' spatial frequency drifting at 0.58 px/frame in direction alpha (as Figure 5A); energy of each
direction unit (its two direction-selective channels), colour = preferred direction (Figure 5 colours); black = sum over
all units. Isotropic and anisotropic cortex (width anisotropy x1 and x2.5, equal peaks, equal numbers). The ratio
sum_iso / sum_aniso is the gain that would make the anisotropic population sum flat (the compensating filter).
New analysis.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_stimulus_filters as SF
from spherical_gabor_bank import LAYOUT_C
out = current_folder+'/Toolbox/Data/work/'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 20})
f0 = SF.F0; alpha = np.radians(np.arange(-180, 180, .5)); PHI_S = np.radians(30)
fg = f0*np.stack([np.cos(PHI_S)*np.cos(alpha), np.cos(PHI_S)*np.sin(alpha), -np.full_like(alpha, np.sin(PHI_S))], 1)


def dir_rgb(a):
    hue = (np.arctan2(-np.sin(a), -np.cos(a))/(2*np.pi)) % 1
    return hsv_to_rgb([hue, 1, .9])


def unit_energy(beta):
    b = SF.bank(beta); E = np.zeros((len(b.mu), len(alpha)))
    for i, (mu, C) in enumerate(zip(b.mu, b.C)):
        E[i] = np.exp(-np.sum((fg-mu)**2, 1)/C[0, 0])+np.exp(-np.sum((-fg-mu)**2, 1)/C[0, 0])
    return E.reshape(16, b.S, -1)[:, LAYOUT_C < 0].sum(1)               # direction-selective units


fig, ax = plt.subplots(2, 3, figsize=(24, 11), gridspec_kw=dict(height_ratios=[2, 1]))
Siso = unit_energy(0).sum(0)
for c, (beta, nm) in enumerate(((0, 'Isotropic'), (1, 'Anisotropic, cat widths (×1)'), (2.5, 'Anisotropic, ×2.5'))):
    E = unit_energy(beta); a = ax[0, c]
    for k in range(16):
        pk = alpha[np.argmax(E[k])]
        a.plot(np.degrees(alpha), E[k], color=dir_rgb(pk), lw=3)
    a.plot(np.degrees(alpha), E.sum(0), 'k', lw=4, label='sum over units')
    a.set(title=nm, xticks=np.arange(-180, 181, 90), xlim=[-180, 180], ylim=[0, 2.6], xlabel='Motion direction (deg)',
          ylabel='energy' if c == 0 else None); a.legend(loc='upper right', fontsize=16)
    b = ax[1, c]
    b.plot(np.degrees(alpha), Siso/E.sum(0), 'k', lw=4)
    b.axhline(1, color='gray', ls=':', lw=2)
    b.set(xticks=np.arange(-180, 181, 90), xlim=[-180, 180], ylim=[.7, 1.3], xlabel='Motion direction (deg)',
          ylabel='sum$_{iso}$ / sum' if c == 0 else None, title='compensating gain')
plt.tight_layout(); fig.savefig(out+'R2_86_figS_direction_energy.png', dpi=70)
for beta in (1, 2.5):
    g = Siso/unit_energy(beta).sum(0)
    print(f'beta {beta}: compensating gain range {g.min():.2f} - {g.max():.2f}; at 0 deg {g[np.argmin(np.abs(np.degrees(alpha)))]:.2f}, '
          f'at 90 deg {g[np.argmin(np.abs(np.degrees(alpha)-90))]:.2f}, at 22.5 deg {g[np.argmin(np.abs(np.degrees(alpha)-22.5))]:.2f}')
