#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Variance of deformations with local neighbourhoods only (Domini, Caudek & Proffitt, 1997, local measure):
for every contour point (anchor), the points within R_px (half the arc length of a 30 deg section) of it, 8 of them
spread by distance rank; def of every triplet; variance; mean +- SEM over anchors, per phase. Three cases:
(1) joined two rings, neighbourhoods over both rings (Revision_R2_figS_domini_region_size, B);
(2) the same two-ring flow, ring 1 only (the other ring's points excluded);
(3) the single-ring stimulus (Figure 5 flows, heeger_fig5_hs_{o}.npz).
Anisotropic and isotropic cortex, vertical and horizontal rotation. Cache heeger_def_variance_local.npy.
Figure: case (1) only (A variance, B vertical - horizontal); cases (2), (3) are cached for reference.
Output: Images/Revision_R2/R2_86_figS_def_variance_local.png. New analysis (cached flows)."""
import os, sys
os.environ.setdefault('K_FIT_TRIM', '10')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_figS_domini_region_size as M
from Revision_R2_heeger_gaussian_flow import ring_points
TR, NP, FRAMES, P, phase, COL = M.TR, M.NP, M.FRAMES, M.P, M.phase, M.COL
plt.rcParams.update({'font.family': 'Arial', 'font.size': 22, 'axes.linewidth': 1.8})
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
    fig, axs = plt.subplots(1, 2, figsize=(26, 9)); ax = axs.reshape(2, 1)
    for c, (case, title) in enumerate(cases):
        a = ax[0, c]
        for o in ('V', 'H'):
            for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
                x = S[(case, o, kind)]; n = np.sum(np.isfinite(x), 1)
                mu, se = np.nanmean(x, 1), np.nanstd(x, 1)/np.sqrt(np.maximum(n, 1))
                if kind == 'A':
                    a.fill_between(phase, np.maximum(mu-se, 1e-7), mu+se, color=COL[o], alpha=.25, lw=0)
                a.plot(phase, mu, ls, color=COL[o], lw=3.5, alpha=1 if kind == 'A' else .55,
                       label=f'{"vertical" if o == "V" else "horizontal"} rotation, {cl}')
        a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], yscale='log'); a.set_title('A', loc='left', fontweight='bold')
        b = ax[1, c]
        for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
            x = S[(case, 'V', kind)]-S[(case, 'H', kind)]; n = np.sum(np.isfinite(x), 1)
            mu, se = np.nanmean(x, 1), np.nanstd(x, 1)/np.sqrt(np.maximum(n, 1))
            if kind == 'A':
                b.fill_between(phase, mu-se, mu+se, color='k', alpha=.2, lw=0)
                lim = np.nanmax(np.abs(mu[fit]))*1.4
                b.text(.02, .04, f'mean (20–160°): {np.nanmean(mu[fit]):+.2g}; mean − SEM > 0 at {np.mean((mu-se)[fit] > 0)*100:.0f}% of phases',
                       transform=b.transAxes, fontsize=15)
                print(f'{case}: V-H mean {np.nanmean(mu[fit]):.3g}, mean/SEM {np.nanmean((mu/se)[fit]):.1f}, '
                      f'>0 {np.mean((mu-se)[fit] > 0)*100:.0f}%, sections {int(np.nanmedian(n[fit]))} | '
                      + ', '.join(f'{o}{k} {np.nanmean(np.nanmean(S[(case, o, k)], 1)[fit]):.3g}' for o in 'VH' for k in 'AU'))
            b.plot(phase, mu, ls, color='k', lw=3.5, label=cl)
        b.axhline(0, color='0.5', lw=1.2, zorder=0)
        b.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], ylim=[-lim, lim], xlabel='Motion phase (deg)'); a.set_xlabel('Motion phase (deg)')
        b.ticklabel_format(axis='y', style='sci', scilimits=(0, 0)); b.set_title('B', loc='left', fontweight='bold')
        for z in (a, b):
            for sp in ('top', 'right'):
                z.spines[sp].set_visible(False)
    ax[0, 0].set_ylabel('Variance of deformations (frame$^{-2}$)'); ax[1, 0].set_ylabel('Vertical − horizontal (frame$^{-2}$)')
    ax[0, 0].legend(fontsize=15, frameon=False, loc='upper center'); ax[1, 0].legend(fontsize=16, frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_figS_def_variance_local.png', dpi=70)
