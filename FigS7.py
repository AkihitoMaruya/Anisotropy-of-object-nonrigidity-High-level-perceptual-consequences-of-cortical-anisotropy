#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary Figure S7. Variability of deformations (Domini, Caudek & Proffitt, 1997) on the joined two rings: for
every contour point, the points within a local neighbourhood (half the arc length of a 30 deg section), def of every
triplet, variance; mean +- SEM over neighbourhoods at each phase. A: vertical (red) and horizontal (blue) rotation,
anisotropic (solid) and isotropic (dotted) cortex. B: vertical - horizontal (mean +- SEM).
Cached results: Toolbox/Data/Rings/heeger_def_variance_local.npy (Toolbox/revision/Revision_R2_figS_def_variance_local.py).
Output: figures_paper/FigS7.pdf."""
import os, sys

import os, sys
os.environ.setdefault('K_FIT_TRIM', '10')
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import figstyle as fs
import Revision_R2_figS_domini_region_size as M
from Revision_R2_heeger_gaussian_flow import ring_points
TR, NP, FRAMES, P, phase, COL = M.TR, M.NP, M.FRAMES, M.P, M.phase, M.COL
fs.style()
W = 20
R_px = np.median([np.linalg.norm(np.diff(TR.two_ring_points('H')[0][FRAMES][:, :W+1], axis=1), axis=-1).sum(1)])/2


def compute():
    cache = P.A.data+'heeger_def_variance_local.npy'
    if os.path.exists(cache):
        return np.load(cache, allow_pickle=True).item()
    S = {}
    two = M.section_spread()                                         # case 1, already computed (240 anchors)
    for o in ('V', 'H'):
        L, _ = TR.two_ring_points(o); Z = TR.flows(o); Ls, _, _ = ring_points(o); Fs = np.load(P.A.data+f'heeger_fig5_hs_{o}.npz')
        for kind in ('A', 'U'):
            S[('two', o, kind)] = two[('B', o, kind)]
            S[('ring1', o, kind)] = M.pixel_neighbour_sections(L[FRAMES][:, :NP], Z[kind][:, :NP], R_px, step=1)
            S[('single', o, kind)] = M.pixel_neighbour_sections(Ls[FRAMES], Fs[f'{kind}_a1.0'], R_px, step=1)
            print(o, kind, 'done', flush=True)
    np.save(cache, S, allow_pickle=True)
    return S


if __name__ == '__main__':
    S = compute()
    fit = (phase > 20) & (phase < 160)
    cases = (('two', 'Two rings'),)
    fig, axs = plt.subplots(1, 2, figsize=(fs.WIDTH, fs.WIDTH*0.42)); ax = axs.reshape(2, 1)
    for c, (case, title) in enumerate(cases):
        a = ax[0, c]
        for o in ('V', 'H'):
            for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
                x = S[(case, o, kind)]; n = np.sum(np.isfinite(x), 1)
                mu, se = np.nanmean(x, 1), np.nanstd(x, 1)/np.sqrt(np.maximum(n, 1))
                if kind == 'A':
                    a.fill_between(phase, np.maximum(mu-se, 1e-7), mu+se, color=COL[o], alpha=.25, lw=0)
                a.plot(phase, mu, ls, color=COL[o], lw=1.27, alpha=1 if kind == 'A' else .55,
                       label=f'{"vertical" if o == "V" else "horizontal"} rotation, {cl}')
        a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], yscale='log', ylim=[5e-4, 5]); fs.letter(a, 'A', x=-.15)
        b = ax[1, c]
        for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
            x = (S[(case, 'V', kind)]-S[(case, 'H', kind)])*1e3; n = np.sum(np.isfinite(x), 1)
            mu, se = np.nanmean(x, 1), np.nanstd(x, 1)/np.sqrt(np.maximum(n, 1))
            if kind == 'A':
                b.fill_between(phase, mu-se, mu+se, color='k', alpha=.2, lw=0)
                lim = np.nanmax(np.abs(mu[fit]))*1.4
                print(f'{case}: V-H mean {np.nanmean(mu[fit])/1e3:.3g}, mean/SEM {np.nanmean((mu/se)[fit]):.1f}, '
                      f'>0 {np.mean((mu-se)[fit] > 0)*100:.0f}%, sections {int(np.nanmedian(n[fit]))} | '
                      + ', '.join(f'{o}{k} {np.nanmean(np.nanmean(S[(case, o, k)], 1)[fit]):.3g}' for o in 'VH' for k in 'AU'))
            b.plot(phase, mu, ls, color='k', lw=1.27, label=cl)
        b.axhline(0, color='0.5', lw=0.436, zorder=0)
        b.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], ylim=[-lim, lim], xlabel='Motion phase (deg)'); a.set_xlabel('Motion phase (deg)')
        fs.letter(b, 'B', x=-.15)
        for z in (a, b):
            for sp in ('top', 'right'):
                z.spines[sp].set_visible(False)
    ax[0, 0].set_ylabel('Variance of deformations (frame$^{-2}$)'); ax[1, 0].set_ylabel('Vertical − horizontal (10$^{-3}$ frame$^{-2}$)')
    ax[0, 0].legend(fontsize=6, frameon=False, loc='upper center'); ax[1, 0].legend(fontsize=6, frameon=False, loc='lower center')
    fig.tight_layout(); fs.save(fig, 'FigS7')
