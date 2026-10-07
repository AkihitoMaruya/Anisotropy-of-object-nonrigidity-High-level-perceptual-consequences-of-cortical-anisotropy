#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variability of deformations (Domini, Caudek & Proffitt, 1997) with larger local regions: a region is a section of
contour with M points; def of every triplet in the region (exact affine flow from three points, their Eq. 2;
triangles with area < MIN_REL x largest squared side skipped); local measure = variance over all triplets of the
region; global = mean over regions, per phase. Contour sections: W consecutive samples (of 240 per ring) along one
ring, sliding by W/2; joint sections: the W samples centred on the joint, M/2 points from each ring. Two rings (D4G,
orthographic; Revision_R2_figS_def_variance_two_rings.py), Figure 5 model flows (isotropic, anisotropic width x2.5),
true rigid rotation and wobble. Output: Images/Revision_R2/R2_86_figS_domini_region_size.png. New analysis.
"""
import os, sys
os.environ.setdefault('K_FIT_TRIM', '10')
import numpy as np
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_figS_def_variance_two_rings as TR
from Revision_R2_heeger_gaussian_flow import FRAMES, NP
P = TR.P
plt.rcParams.update({'font.family': 'Arial', 'font.size': 22, 'axes.linewidth': 1.8})
M, MIN_REL = 12, 0.01
WS = [30, 60, 120, 240]                                           # section length (samples; 240 = whole ring)
COL = {'V': '#c0392b', 'H': '#2471a3'}
phase = TR.phase


def regions(W):
    cont = []
    for r in (0, 1):
        for s in range(0, NP, max(W//2, 1)):
            idx = (s+np.round(np.linspace(0, W, M, endpoint=W >= NP and False)).astype(int)) % NP
            cont.append(r*NP+np.unique(idx))
    half = np.round(np.linspace(-W/2, W/2, M//2)).astype(int) % NP
    joint = [np.concatenate([half, NP+half])]
    return cont, joint


def region_var(L, vel, regs):
    """(frames, regions): variance of def over all non-degenerate triplets of each region."""
    out = np.full((len(L), len(regs)), np.nan)
    for ri, reg in enumerate(regs):
        tri = np.array(list(combinations(reg, 3)))
        for t, (Lt, vt) in enumerate(zip(L, vel)):
            x, y = Lt[tri, 1], -Lt[tri, 0]
            area = 0.5*np.abs((x[:, 1]-x[:, 0])*(y[:, 2]-y[:, 0])-(x[:, 2]-x[:, 0])*(y[:, 1]-y[:, 0]))
            side = np.max([(x[:, a]-x[:, b])**2+(y[:, a]-y[:, b])**2 for a, b in ((0, 1), (0, 2), (1, 2))], 0)
            ok = area >= MIN_REL*side
            if ok.sum() < 3:
                continue
            Mx = np.stack([np.ones(ok.sum()*3).reshape(-1, 3), x[ok], y[ok]], -1)
            a = np.linalg.solve(Mx, vt[tri[ok], 0][..., None])[..., 0]; b = np.linalg.solve(Mx, vt[tri[ok], 1][..., None])[..., 0]
            out[t, ri] = np.var(np.hypot(a[:, 1]-b[:, 2], a[:, 2]+b[:, 1]))
    return out


if __name__ == '__main__':
    data = {}
    for o in ('V', 'H'):
        L, vt = TR.two_ring_points(o); Lw, vw = TR.two_ring_points_k(o, 1.0); Z = TR.flows(o)
        data[o] = {'rigid': (L[FRAMES], vt[FRAMES]), 'wobble': (Lw[FRAMES], vw[FRAMES]),
                   'A': (L[FRAMES], Z['A']), 'U': (L[FRAMES], Z['U'])}
    fit = (phase > 20) & (phase < 160); mid = np.abs(phase-90) < 10
    R = {}
    for W in WS:
        cont, joint = regions(W)
        for o in ('V', 'H'):
            for kind, (L, v) in data[o].items():
                if o == 'H' and kind in ('rigid', 'wobble'):
                    continue
                c = np.nanmean(region_var(L, v, cont), 1); j = region_var(L, v, joint)[:, 0]
                R[(W, o, kind)] = (c, j)
                print(f'W {W:3d} ({W*360/NP:.0f} deg) {o} {kind:6s}: contour {np.nanmean(c[fit]):.3g} (90: {np.nanmean(c[mid]):.3g}) | '
                      f'joint {np.nanmean(j[fit]):.3g} (90: {np.nanmean(j[mid]):.3g})', flush=True)
    np.save(P.A.data+'heeger_domini_region_size.npy', R, allow_pickle=True)
    fig, ax = plt.subplots(1, 2, figsize=(24, 8), sharey=True)
    xs = np.array(WS)*360/NP
    for a, part, title in ((ax[0], 1, 'Sections at the joint'), (ax[1], 0, 'Sections along each ring')):
        for o in ('V', 'H'):
            for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
                a.plot(xs, [np.nanmean(R[(W, o, kind)][part][fit]) for W in WS], ls, marker='o', ms=9, color=COL[o], lw=3.5,
                       label=f'{"vertical" if o == "V" else "horizontal"} rotation, {cl}')
        a.plot(xs, [np.nanmean(R[(W, 'V', 'wobble')][part][fit]) for W in WS], '--', marker='s', color='k', lw=2.5, label='wobble (true velocity)')
        a.set(xlabel='Section length (deg of ring)', xticks=xs, yscale='log'); a.set_title(title, loc='left', fontweight='bold')
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel('Mean variance of deformations (frame$^{-2}$)')
    ax[1].legend(fontsize=15, frameon=False)
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_figS_domini_region_size.png', dpi=80)


def plot_by_phase(R):
    """Grid: rows joint / contour sections, columns section length; curves over motion phase.
    Output: Images/Revision_R2/R2_86_figS_domini_region_size_phase.png"""
    fig, ax = plt.subplots(2, len(WS), figsize=(9*len(WS), 15), sharey='row')
    for c, W in enumerate(WS):
        for r, (part, rowlab) in enumerate(((1, 'Sections at the joint'), (0, 'Sections along each ring'))):
            a = ax[r, c]
            for o in ('V', 'H'):
                for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
                    a.plot(phase, R[(W, o, kind)][part], ls, color=COL[o], lw=3, alpha=.55 if kind == 'U' else 1, label=f'{"vertical" if o == "V" else "horizontal"} rotation, {cl}')
            a.plot(phase, R[(W, 'V', 'wobble')][part], '--', color='k', lw=2.5, label='wobble (true velocity)')
            if part == 1:
                a.plot(phase, R[(W, 'V', 'rigid')][part], '--', color='0.55', lw=2.5, label='rigid rotation (true velocity)')
            a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], yscale='log')
            if r == 0:
                a.set_title(f'Section {W*360/NP:.0f}° of ring', fontweight='bold')
            if r == 1:
                a.set_xlabel('Motion phase (deg)')
            if c == 0:
                a.set_ylabel(f'{rowlab}\nvariance of deformations (frame$^{{-2}}$)')
            for sp in ('top', 'right'):
                a.spines[sp].set_visible(False)
    h, l = ax[0, -1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=17, frameon=False)
    ax[1, 0].text(178, ax[1, 0].get_ylim()[0]*2, 'rigid rotation: ≈ 0', ha='right', va='bottom', fontsize=14, color='0.4')
    fig.tight_layout(rect=[0, .07, 1, 1]); fig.savefig(P.A.out+'R2_86_figS_domini_region_size_phase.png', dpi=70)


def overlapping_sections(W, m=8, step=4, joint_only=False):
    """Sections of W samples centred on every `step`-th point of each ring (overlapping); a section whose centre is
    within W/2 of the joint (index 0) also takes the other ring's points over the same arc (m/2 from each ring)."""
    regs = []
    for r in (0, 1):
        for c in range(0, NP, step):
            near = min(c, NP-c) <= W/2
            if joint_only and (not near or r == 1):                  # joint sections only (once: they span both rings)
                continue
            k = m//2 if near else m
            idx = (c+np.round(np.linspace(-W/2, W/2, k)).astype(int)) % NP
            regs.append(np.concatenate([idx, NP+idx]) if near else r*NP+idx)
    return regs


def overall_by_phase(W_list=(10, 20, 30, 60)):
    """Overall measure per phase (mean of the local variances over all overlapping sections), for each section
    length. Output: Images/Revision_R2/R2_86_figS_domini_overlap_phase.png"""
    data = {}
    for o in ('V', 'H'):
        L, vt = TR.two_ring_points(o); Lw, vw = TR.two_ring_points_k(o, 1.0); Z = TR.flows(o)
        data[o] = {'rigid': (L[FRAMES], vt[FRAMES]), 'wobble': (Lw[FRAMES], vw[FRAMES]), 'A': (L[FRAMES], Z['A']), 'U': (L[FRAMES], Z['U'])}
    G = {}
    fig, ax = plt.subplots(1, len(W_list), figsize=(9*len(W_list), 8.5), sharey=True)
    fit = (phase > 20) & (phase < 160)
    for a, W in zip(ax, W_list):
        regs = overlapping_sections(W)
        for o in ('V', 'H'):
            for kind in (('A', 'U', 'rigid', 'wobble') if o == 'V' else ('A', 'U')):
                G[(W, o, kind)] = np.nanmean(region_var(*data[o][kind], regs), 1)
        for o in ('V', 'H'):
            for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
                a.plot(phase, G[(W, o, kind)], ls, color=COL[o], lw=3, alpha=.55 if kind == 'U' else 1, label=f'{"vertical" if o == "V" else "horizontal"} rotation, {cl}')
        a.plot(phase, G[(W, 'V', 'wobble')], '--', color='k', lw=2.5, label='wobble (true velocity)')
        a.plot(phase, G[(W, 'V', 'rigid')], '--', color='0.55', lw=2.5, label='rigid rotation (true velocity)')
        a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], yscale='log', xlabel='Motion phase (deg)')
        a.set_title(f'Sections of {W*360/NP:.0f}°', fontweight='bold')
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        print(f'W {W*360/NP:.0f} deg: ' + ', '.join(f'{o}{k} {np.nanmean(G[(W, o, k)][fit]):.3g}' for (w, o, k) in G if w == W), flush=True)
    ax[0].set_ylabel('Mean local variance of deformations (frame$^{-2}$)')
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=17, frameon=False)
    fig.tight_layout(rect=[0, .12, 1, 1]); fig.savefig(P.A.out+'R2_86_figS_domini_overlap_phase.png', dpi=70)
    np.save(P.A.data+'heeger_domini_overlap.npy', G, allow_pickle=True)
    return G


def single_figure(W=20):
    """One panel: overall variability of deformations (overlapping sections of W samples) against motion phase.
    Output: Images/Revision_R2/R2_86_figS_def_variance_phase.png"""
    G = np.load(P.A.data+'heeger_domini_overlap.npy', allow_pickle=True).item()
    fig, a = plt.subplots(figsize=(14, 9))
    for o in ('V', 'H'):
        for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
            a.plot(phase, G[(W, o, kind)], ls, color=COL[o], lw=3.5, alpha=.55 if kind == 'U' else 1, label=f'{"vertical" if o == "V" else "horizontal"} rotation, {cl}')
    a.plot(phase, G[(W, 'V', 'wobble')], '--', color='k', lw=2.5, label='wobble (true velocity)')
    a.plot(phase, G[(W, 'V', 'rigid')], '--', color='0.55', lw=2.5, label='rigid rotation (true velocity)')
    a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], yscale='log', xlabel='Motion phase (deg)',
          ylabel='Variance of deformations (frame$^{-2}$)')
    a.legend(fontsize=15, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=3)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_figS_def_variance_phase.png', dpi=80, bbox_inches='tight')


def joint_figure(W=20, step=1):
    """One panel: variability of deformations against motion phase from the overlapping sections around the joint
    only (centres within W/2 of the joint, m/2 points from each ring, sliding by `step` samples).
    Output: Images/Revision_R2/R2_86_figS_def_variance_phase.png"""
    regs = overlapping_sections(W, step=step, joint_only=True)
    G = {}
    for o in ('V', 'H'):
        L, vt = TR.two_ring_points(o); Lw, vw = TR.two_ring_points_k(o, 1.0); Z = TR.flows(o)
        d = {'A': (L[FRAMES], Z['A']), 'U': (L[FRAMES], Z['U'])}
        for kind, (Lk, vk) in d.items():
            G[(o, kind)] = np.nanmean(region_var(Lk, vk, regs), 1)
    fig, a = plt.subplots(figsize=(14, 9))
    for o in ('V', 'H'):
        for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
            a.plot(phase, G[(o, kind)], ls, color=COL[o], lw=3.5, alpha=.55 if kind == 'U' else 1, label=f'{"vertical" if o == "V" else "horizontal"} rotation, {cl}')
    a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], yscale='log', xlabel='Motion phase (deg)',
          ylabel='Variance of deformations at the joint (frame$^{-2}$)')
    a.legend(fontsize=15, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_figS_def_variance_phase.png', dpi=80, bbox_inches='tight')
    fit = (phase > 20) & (phase < 160); mid = np.abs(phase-90) < 10
    for k, v in G.items():
        print(k, f'20-160: {np.nanmean(v[fit]):.3g}  near 90: {np.nanmean(v[mid]):.3g}')
    return G


def paired_sections(W, m=8, step=2):
    """Sections over the whole display that pair the two rings at the same image x (vertical rotation: same y):
    the same contour range c +- W/2 on both rings (same index = mirror points, same depth), m/2 points from each."""
    regs = []
    for c in range(0, NP, step):
        idx = (c+np.round(np.linspace(-W/2, W/2, m//2)).astype(int)) % NP
        regs.append(np.concatenate([idx, NP+idx]))
    return regs


def paired_figure(W=20):
    """One panel: variability of deformations against motion phase, mean over the paired sections around the whole
    display. Output: Images/Revision_R2/R2_86_figS_def_variance_phase_paired.png"""
    regs = paired_sections(W)
    G = {}
    for o in ('V', 'H'):
        L, vt = TR.two_ring_points(o); Lw, vw = TR.two_ring_points_k(o, 1.0); Z = TR.flows(o)
        d = {'A': (L[FRAMES], Z['A']), 'U': (L[FRAMES], Z['U'])}
        if o == 'V':
            d.update({'rigid': (L[FRAMES], vt[FRAMES]), 'wobble': (Lw[FRAMES], vw[FRAMES])})
        for kind, (Lk, vk) in d.items():
            G[(o, kind)] = np.nanmean(region_var(Lk, vk, regs), 1)
    fig, a = plt.subplots(figsize=(14, 9))
    for o in ('V', 'H'):
        for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
            a.plot(phase, G[(o, kind)], ls, color=COL[o], lw=3.5, alpha=.55 if kind == 'U' else 1, label=f'{"vertical" if o == "V" else "horizontal"} rotation, {cl}')
    a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], yscale='log', xlabel='Motion phase (deg)',
          ylabel='Variance of deformations (frame$^{-2}$)')
    a.legend(fontsize=15, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_figS_def_variance_phase_paired.png', dpi=80, bbox_inches='tight')
    fit = (phase > 20) & (phase < 160); mid = np.abs(phase-90) < 10
    for k, v in G.items():
        print(k, f'20-160: {np.nanmean(v[fit]):.3g}  near 90: {np.nanmean(v[mid]):.3g}')
    return G


def pixel_neighbour_var(L, vel, R_px=25.0, m=8, step=4):
    """B: local neighbours in pixel location. Per frame, for every `step`-th contour point (both rings) the points
    within R_px pixels of it (either ring), m of them spread by distance rank; variance of def over all triplets;
    mean over anchors."""
    out = np.full(len(L), np.nan)
    for t, (Lt, vt) in enumerate(zip(L, vel)):
        vs = []
        for a in range(0, len(Lt), step):
            d = np.linalg.norm(Lt-Lt[a], axis=1); nb = np.argsort(d)[:np.sum(d <= R_px)]
            if len(nb) < m:
                continue
            reg = nb[np.round(np.linspace(0, len(nb)-1, m)).astype(int)]
            v = region_var(Lt[None], vt[None], [reg])[0, 0]
            if np.isfinite(v):
                vs.append(v)
        out[t] = np.mean(vs) if vs else np.nan
    return out


def methods_figure(W=20):
    """A: global triplets; B: local neighbours in pixel location; C: symmetric pairs, sections pairing each bottom-ring
    point with its mirror point on the top ring (same x for horizontal rotation, same y for vertical rotation). Output: Images/Revision_R2/R2_86_figS_def_variance_methods.png"""
    regsC = paired_sections(W)
    R_px = np.median([np.linalg.norm(np.diff(TR.two_ring_points('H')[0][FRAMES][:, :W+1], axis=1), axis=-1).sum(1)])/2
    G = {}
    for o in ('V', 'H'):
        L, vt = TR.two_ring_points(o); Lw, vw = TR.two_ring_points_k(o, 1.0); Z = TR.flows(o)
        d = {'A': (L[FRAMES], Z['A']), 'U': (L[FRAMES], Z['U'])}
        if o == 'V':
            d.update({'rigid': (L[FRAMES], vt[FRAMES]), 'wobble': (Lw[FRAMES], vw[FRAMES])})
        for kind, (Lk, vk) in d.items():
            G[('A', o, kind)] = np.array([np.var(x) for x in TR.triplet_def(Lk, vk)])
            G[('B', o, kind)] = pixel_neighbour_var(Lk, vk, R_px)
            G[('C', o, kind)] = np.nanmean(region_var(Lk, vk, regsC), 1)
            print(o, kind, 'done', flush=True)
    np.save(P.A.data+'heeger_def_variance_methods.npy', G, allow_pickle=True)
    plot_methods(G)
    return G


def plot_methods(G=None, refs=False, spread=True):
    """Figure of methods_figure from its cache; refs: also draw true rotation and wobble."""
    if G is None:
        G = np.load(P.A.data+'heeger_def_variance_methods.npy', allow_pickle=True).item()
    if spread:                                                         # B, C from the 240 sections each
        Ssec = section_spread(); G = dict(G)
        for (mth, o, kind), x in Ssec.items():
            G[(mth, o, kind)] = np.nanmean(x, 1)
    titles = {'A': 'A   Global triplets', 'B': 'B   Local neighbours',
              'C': 'C   Symmetric pairs'}
    import Revision_R2_figS_section_example as SE
    import sys as _s
    fig = plt.figure(figsize=(30, 24))
    gs = fig.add_gridspec(3, 3, height_ratios=[.55, 1, 1])
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in (1, 2)])
    for c, mth, t in ((0, 'A', 'Global triplets'), (1, 'B', 'Local-neighbour sections'), (2, 'C', 'Symmetric-pair sections')):
        sa = fig.add_subplot(gs[0, c]); SE.draw_schematic(sa, mth, _s.modules[__name__])
        sa.set_title(t, loc='left', fontweight='bold', pad=34)
    ax = axs[0]
    for a in ax[1:]:
        a.sharey(ax[0])
    fit = (phase > 20) & (phase < 160)
    for a, mth, lab in zip(axs[1], 'ABC', 'DEF'):                       # vertical minus horizontal
        for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
            d = G[(mth, 'V', kind)]-G[(mth, 'H', kind)]
            a.plot(phase, d, ls, color='k', lw=3.5, label=cl)
        d = G[(mth, 'V', 'A')]-G[(mth, 'H', 'A')]
        if spread and mth in 'BC':                                     # paired differences across sections, mean +- SEM
            x = Ssec[(mth, 'V', 'A')]-Ssec[(mth, 'H', 'A')]; n = np.sum(np.isfinite(x), 1)
            mu, se = np.nanmean(x, 1), np.nanstd(x, 1)/np.sqrt(np.maximum(n, 1))
            a.fill_between(phase, mu-se, mu+se, color='k', alpha=.2, lw=0)
            print(f'{mth} V-H: mean/SEM (20-160) {np.nanmean((mu/se)[fit]):.1f}; phases with mean-SEM > 0: {np.mean((mu-se)[fit] > 0)*100:.0f}%')
        lim = np.nanmax(np.abs(d[fit]))*1.3
        a.axhline(0, color='0.5', lw=1.2, zorder=0)
        a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], ylim=[-lim, lim], xlabel='Motion phase (deg)')
        a.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        a.set_title(f'{lab}   {titles[mth][4:]}', loc='left', fontweight='bold')
        a.text(.02, .04, f'mean (20–160°): {np.nanmean(d[fit]):+.2g}  ({np.nanmean(d[fit] > 0)*100:.0f}% of phases > 0)',
               transform=a.transAxes, fontsize=16)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    axs[1, 0].set_ylabel('Vertical − horizontal (frame$^{-2}$)')
    axs[1, 2].legend(fontsize=16, frameon=False, loc='upper right')
    for a, mth in zip(ax, 'ABC'):
        if spread and mth in 'BC':                                     # mean +- SEM across sections
            for o in ('V', 'H'):
                x = Ssec[(mth, o, 'A')]; n = np.sum(np.isfinite(x), 1)
                mu, se = np.nanmean(x, 1), np.nanstd(x, 1)/np.sqrt(np.maximum(n, 1))
                a.fill_between(phase, np.maximum(mu-se, 1e-9), mu+se, color=COL[o], alpha=.25, lw=0)
        for o in ('V', 'H'):
            for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
                a.plot(phase, G[(mth, o, kind)], ls, color=COL[o], lw=3.5, alpha=.55 if kind == 'U' else 1, label=f'{"vertical" if o == "V" else "horizontal"} rotation, {cl}')
        if refs:
            a.plot(phase, G[(mth, 'V', 'wobble')], '--', color='k', lw=1.8, label='wobble (true velocity)')
            a.plot(phase, G[(mth, 'V', 'rigid')], '--', color='0.6', lw=1.8, label='rigid rotation (true velocity)')
        a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], yscale='log')
        a.set_title(titles[mth], loc='left', fontweight='bold')
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        print(mth, ', '.join(f'{o}{k} {np.nanmean(G[(mth, o, k)][fit]):.3g}' for (m_, o, k) in G if m_ == mth))
    ax[0].set_ylabel('Variance of deformations (frame$^{-2}$)')
    ax[0].legend(fontsize=15, frameon=False, loc='upper center')
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_figS_def_variance_methods.png', dpi=70)


def pixel_neighbour_sections(L, vel, R_px=25.0, m=8, step=4):
    """As pixel_neighbour_var, but returns the local variance of every section: (frames, anchors); a section uses up to m
    points within R_px (at least 4), nan where fewer."""
    out = np.full((len(L), len(range(0, L.shape[1], step))), np.nan)
    for t, (Lt, vt) in enumerate(zip(L, vel)):
        for ai, a in enumerate(range(0, len(Lt), step)):
            d = np.linalg.norm(Lt-Lt[a], axis=1); nb = np.argsort(d)[:np.sum(d <= R_px)]
            if len(nb) < 4:                                              # up to m points, at least 4
                continue
            k = min(m, len(nb))
            out[t, ai] = region_var(Lt[None], vt[None], [nb[np.round(np.linspace(0, len(nb)-1, k)).astype(int)]])[0, 0]
    return out


def section_spread(W=20):
    """Per-section local variances for B (pixel neighbours: an anchor at every other of the 2 x 240 contour points)
    and C (symmetric pairs with sections of equal 2-D contour length, symmetric_equal_arc), 240 sections each, model
    flows; cached in heeger_def_variance_sections240_arc.npy as {(method, o, kind): (frames, sections)}."""
    cache = P.A.data+'heeger_def_variance_sections240_arc.npy'
    old = P.A.data+'heeger_def_variance_sections240.npy'                 # B is unchanged: reuse it
    if os.path.exists(old):
        S = {k: v for k, v in np.load(old, allow_pickle=True).item().items() if k[0] == 'B'}
        for o in ('V', 'H'):
            L, _ = TR.two_ring_points(o); Z = TR.flows(o)
            for kind in ('A', 'U'):
                S[('C', o, kind)] = symmetric_equal_arc(L[FRAMES], Z[kind])
        np.save(cache, S, allow_pickle=True)
        return S
    if os.path.exists(cache):
        return np.load(cache, allow_pickle=True).item()
    regsC = paired_sections(W, step=1)
    R_px = np.median([np.linalg.norm(np.diff(TR.two_ring_points('H')[0][FRAMES][:, :W+1], axis=1), axis=-1).sum(1)])/2
    S = {}
    for o in ('V', 'H'):
        L, _ = TR.two_ring_points(o); Z = TR.flows(o)
        for kind in ('A', 'U'):
            S[('B', o, kind)] = pixel_neighbour_sections(L[FRAMES], Z[kind], R_px, step=2)
            S[('C', o, kind)] = symmetric_equal_arc(L[FRAMES], Z[kind])
            print(o, kind, 'sections done', flush=True)
    np.save(cache, S, allow_pickle=True)
    return S


def symmetric_pair_sizes(Ws=(6, 10, 20, 40), step=1):
    """C (symmetric pairs) for several section lengths W (samples; 240 = whole ring), sections sliding by `step`:
    per-section local variances for the anisotropic cortex, vertical and horizontal; cached in
    heeger_def_variance_sizes.npy as {(W, o): (frames, sections)}."""
    cache = P.A.data+'heeger_def_variance_sizes.npy'
    S = np.load(cache, allow_pickle=True).item() if os.path.exists(cache) else {}
    for W in Ws:
        regs = [np.unique(r) for r in paired_sections(W, step=step)]
        for o in ('V', 'H'):
            if (W, o) in S:
                continue
            L, _ = TR.two_ring_points(o); Z = TR.flows(o)
            S[(W, o)] = region_var(L[FRAMES], Z['A'], regs); np.save(cache, S, allow_pickle=True)
            print(f'W {W} {o} done', flush=True)
    fit = (phase > 20) & (phase < 160)
    for W in Ws:
        x = S[(W, 'V')]-S[(W, 'H')]; n = np.sum(np.isfinite(x), 1)
        mu, se = np.nanmean(x, 1), np.nanstd(x, 1)/np.sqrt(np.maximum(n, 1))
        print(f'section {W*360/NP:4.0f} deg ({int(np.nanmedian(n))} sections): V-H mean {np.nanmean(mu[fit]):.3g}, '
              f'SEM {np.nanmean(se[fit]):.3g}, mean/SEM {np.nanmean((mu/se)[fit]):.1f}, phases with mean-SEM > 0: '
              f'{np.mean((mu-se)[fit] > 0)*100:.0f}% | V {np.nanmean(np.nanmean(S[(W, "V")], 1)[fit]):.3g}, H {np.nanmean(np.nanmean(S[(W, "H")], 1)[fit]):.3g}')
    return S


def pixel_neighbour_sizes(Ws=(6, 10, 20, 40), step=2):
    """B (local neighbours in pixel location) for the same section lengths as symmetric_pair_sizes and the same number
    of sections (anchors every `step`-th of the 2 x 240 contour points = 240 sections); radius = half the arc length
    of W samples. Anisotropic cortex, vertical and horizontal; cached in heeger_def_variance_sizes_B.npy."""
    cache = P.A.data+'heeger_def_variance_sizes_B.npy'
    S = np.load(cache, allow_pickle=True).item() if os.path.exists(cache) else {}
    arc = np.median(np.linalg.norm(np.diff(TR.two_ring_points('H')[0][FRAMES], axis=1), axis=-1))   # px per sample
    for W in Ws:
        for o in ('V', 'H'):
            if (W, o) in S:
                continue
            L, _ = TR.two_ring_points(o); Z = TR.flows(o)
            S[(W, o)] = pixel_neighbour_sections(L[FRAMES], Z['A'], R_px=arc*W/2, step=step); np.save(cache, S, allow_pickle=True)
            print(f'W {W} {o} done', flush=True)
    fit = (phase > 20) & (phase < 160)
    for W in Ws:
        x = S[(W, 'V')]-S[(W, 'H')]; n = np.sum(np.isfinite(x), 1)
        mu, se = np.nanmean(x, 1), np.nanstd(x, 1)/np.sqrt(np.maximum(n, 1))
        print(f'B section {W*360/NP:4.0f} deg ({int(np.nanmedian(n))} sections): V-H mean {np.nanmean(mu[fit]):.3g}, '
              f'SEM {np.nanmean(se[fit]):.3g}, mean/SEM {np.nanmean((mu/se)[fit]):.1f}, phases with mean-SEM > 0: '
              f'{np.mean((mu-se)[fit] > 0)*100:.0f}% | V {np.nanmean(np.nanmean(S[(W, "V")], 1)[fit]):.3g}, H {np.nanmean(np.nanmean(S[(W, "H")], 1)[fit]):.3g}')
    return S


def plot_sizes(Ws=(6, 10, 20, 40)):
    """Grid: rows B (local neighbours) and C (symmetric pairs), columns section length; vertical - horizontal
    (anisotropic cortex), mean +- SEM across sections, against motion phase.
    Output: Images/Revision_R2/R2_86_figS_def_variance_sizes.png"""
    SB = np.load(P.A.data+'heeger_def_variance_sizes_B.npy', allow_pickle=True).item()
    SC = np.load(P.A.data+'heeger_def_variance_sizes.npy', allow_pickle=True).item()
    fit = (phase > 20) & (phase < 160)
    fig, ax = plt.subplots(2, len(Ws), figsize=(8*len(Ws), 13), sharey='row')
    for r, (S, lab) in enumerate(((SB, 'B   Local neighbours'), (SC, 'C   Symmetric pairs'))):
        for c, W in enumerate(Ws):
            a = ax[r, c]
            x = S[(W, 'V')]-S[(W, 'H')]; n = np.sum(np.isfinite(x), 1)
            mu, se = np.nanmean(x, 1), np.nanstd(x, 1)/np.sqrt(np.maximum(n, 1))
            a.fill_between(phase, mu-se, mu+se, color='k', alpha=.2, lw=0)
            a.plot(phase, mu, 'k', lw=3)
            a.axhline(0, color='0.5', lw=1.2, zorder=0)
            a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], ylim=[-.004, .009])
            a.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            if r == 0:
                a.set_title(f'Sections of {W*360/NP:.0f}°', fontweight='bold')
            if r == 1:
                a.set_xlabel('Motion phase (deg)')
            if c == 0:
                a.set_ylabel(f'{lab}\nvertical − horizontal (frame$^{{-2}}$)')
            a.text(.03, .95, f'{int(np.nanmedian(n))} sections\nabove 0: {np.mean((mu-se)[fit] > 0)*100:.0f}% of phases',
                   transform=a.transAxes, va='top', fontsize=15)
            for sp in ('top', 'right'):
                a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_figS_def_variance_sizes.png', dpi=70)


def plot_sizes_global(Ws=(6, 10, 20, 40)):
    """Grid: rows B (local neighbours) and C (symmetric pairs), columns section length; global measure (mean of the
    local variances over sections) +- SEM across sections, vertical (red) and horizontal (blue) rotation, anisotropic
    cortex, against motion phase. Output: Images/Revision_R2/R2_86_figS_def_variance_sizes_global.png"""
    SB = np.load(P.A.data+'heeger_def_variance_sizes_B.npy', allow_pickle=True).item()
    SC = np.load(P.A.data+'heeger_def_variance_sizes.npy', allow_pickle=True).item()
    fig, ax = plt.subplots(2, len(Ws), figsize=(8*len(Ws), 13), sharey=True)
    for r, (S, lab) in enumerate(((SB, 'B   Local neighbours'), (SC, 'C   Symmetric pairs'))):
        for c, W in enumerate(Ws):
            a = ax[r, c]
            for o in ('V', 'H'):
                x = S[(W, o)]; n = np.sum(np.isfinite(x), 1)
                mu, se = np.nanmean(x, 1), np.nanstd(x, 1)/np.sqrt(np.maximum(n, 1))
                a.fill_between(phase, np.maximum(mu-se, 1e-7), mu+se, color=COL[o], alpha=.25, lw=0)
                a.plot(phase, mu, color=COL[o], lw=3, label='vertical rotation' if o == 'V' else 'horizontal rotation')
            a.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180], yscale='log', ylim=[1e-4, 1])
            if r == 0:
                a.set_title(f'Sections of {W*360/NP:.0f}°', fontweight='bold')
            if r == 1:
                a.set_xlabel('Motion phase (deg)')
            if c == 0:
                a.set_ylabel(f'{lab}\nvariance of deformations (frame$^{{-2}}$)')
            for sp in ('top', 'right'):
                a.spines[sp].set_visible(False)
    ax[0, -1].legend(frameon=False, loc='upper center')
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_figS_def_variance_sizes_global.png', dpi=70)


def symmetric_equal_arc(L, vel, frac=20/240, n_sec=240, m=8):
    """C with sections of equal 2-D contour length: per frame, arc length s along each ring from the joint (index 0);
    n_sec section centres evenly spaced in s; each section spans frac x the ring's image perimeter, m/2 points evenly
    spaced in s on each ring (the same s on both rings = mirror points), positions and velocities interpolated along
    the contour; variance of def over all triplets of the m points. Returns (frames, n_sec)."""
    from itertools import combinations
    tri = np.array(list(combinations(range(m), 3)))
    out = np.full((len(L), n_sec), np.nan)
    for t, (Lt, vt) in enumerate(zip(L, vel)):
        P_, V_ = [], []
        for r in (0, 1):
            p = Lt[r*NP:(r+1)*NP]; v = vt[r*NP:(r+1)*NP]
            pc = np.vstack([p, p[:1]]); vc = np.vstack([v, v[:1]])
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pc, axis=0), axis=1))]; S = s[-1]
            c = np.arange(n_sec)/n_sec*S
            q = (c[:, None]+np.linspace(-frac*S/2, frac*S/2, m//2)[None]) % S        # (n_sec, m/2)
            P_.append(np.stack([np.interp(q, s, pc[:, k]) for k in (0, 1)], -1))
            V_.append(np.stack([np.interp(q, s, vc[:, k]) for k in (0, 1)], -1))
        Pt = np.concatenate(P_, 1); Vt = np.concatenate(V_, 1)              # (n_sec, m, 2)
        x, y = Pt[..., 1][:, tri], -Pt[..., 0][:, tri]                      # (n_sec, 56, 3)
        u, v = Vt[..., 0][:, tri], Vt[..., 1][:, tri]
        area = 0.5*np.abs((x[..., 1]-x[..., 0])*(y[..., 2]-y[..., 0])-(x[..., 2]-x[..., 0])*(y[..., 1]-y[..., 0]))
        side = np.max([(x[..., a]-x[..., b])**2+(y[..., a]-y[..., b])**2 for a, b in ((0, 1), (0, 2), (1, 2))], 0)
        ok = area >= MIN_REL*side
        Mx = np.stack([np.ones_like(x), x, y], -1); Mx[~ok] = np.eye(3)
        a = np.linalg.solve(Mx, u[..., None])[..., 0]; b = np.linalg.solve(Mx, v[..., None])[..., 0]
        de = np.where(ok, np.hypot(a[..., 1]-b[..., 2], a[..., 2]+b[..., 1]), np.nan)
        good = np.sum(ok, 1) >= 3
        out[t, good] = np.nanvar(de[good], 1)
    return out
