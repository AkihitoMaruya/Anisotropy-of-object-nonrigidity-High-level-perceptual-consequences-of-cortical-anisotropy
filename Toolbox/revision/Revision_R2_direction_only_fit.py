#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Magnitude-free rotation (k = 0) vs wobble (k = 1) fits of the model optic flows on the rotating D4G ring.
  Method 2: every flow vector (and every template velocity) is normalised to unit length; div, curl, def are then
            the contour integrals / enclosed area; k is fitted by least squares over frames (first/last 4 excluded).
  Method 3: per-point direction matching (cosine similarity between flow and template direction, averaged over
            points and frames; best k = highest similarity), as in Figure 5F/G.
Flows: normal flow, Gaussian Heeger isotropic, anisotropic x1 and x4 (layout C, raw normalisation, averaged across
the contour). New analysis; reads existing code and data only.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_fig6B_per_area_flows as P
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES

C = P.C
KG = np.round(np.arange(0, 1.5001, 0.01), 2)


def unit(U, V):
    n = np.hypot(U, V)
    n = np.where(n > 1e-12, n, np.inf)
    return U/n, V/n


def tmpl_vel(k, o):                                   # ring coordinates, per radian
    return C.analytic_velocity(k, o)


def inv_unit(U, V, o):                                # method 2
    return P.per_area(*unit(U, V), o)


def flow_phys(fl):
    return -fl[..., 0], -fl[..., 1]                   # screen -> ring coordinates (scale irrelevant here)


T2 = {o: np.array([inv_unit(*tmpl_vel(k, o), o) for k in KG]) for o in ('H', 'V')}
TU = {o: np.array([np.stack(unit(*tmpl_vel(k, o))) for k in KG]) for o in ('H', 'V')}   # (k, 2, frames, points)


def fit2(inv, o):
    cost = np.sum((inv[None, :, P.FIT]-T2[o][:, :, P.FIT])**2, (1, 2))
    return KG[cost.argmin()]


def fit3(U, V, o):
    u, v = unit(U, V)
    ok = np.isfinite(u) & (np.hypot(U, V) > 1e-9)
    sim = []
    for T in TU[o]:
        c = (u*T[0]+v*T[1])[P.FIT][ok[P.FIT]]
        sim.append(c.mean())
    sim = np.array(sim)
    return KG[sim.argmax()], sim


if __name__ == '__main__':
    runs = (('true', 'true velocity', 'tab:green'), ('normal', 'normal flow', 'k'), ('iso', 'isotropic', 'tab:blue'),
            ('aniso1', 'anisotropic ×1', 'tab:orange'), ('aniso4', 'anisotropic ×4', 'tab:red'))
    res, sims, invs = {}, {}, {}
    for o in ('H', 'V'):
        Fl = np.load(P.A.data+f'heeger_layoutC_across_{o}.npz')
        L, vt, n = ring_points(o)
        vt, n = vt[FRAMES], n[FRAMES]
        vn = np.sum(vt*n, -1, keepdims=True)*n
        for key, lab, col in runs:
            fl = vt if key == 'true' else vn if key == 'normal' else Fl[key].mean(0)
            U, V = flow_phys(fl)
            inv = inv_unit(U, V, o)
            k2 = fit2(inv, o); k3, s = fit3(U, V, o)
            res[(o, key)] = (k2, k3); sims[(o, key)] = s; invs[(o, key)] = inv
    print(f'{"flow":16s} | method 2 (unit vectors, invariants): k_H  k_V | method 3 (direction matching): k_H  k_V')
    for key, lab, _ in runs:
        print(f'{lab:16s} |   {res[("H", key)][0]:.2f}  {res[("V", key)][0]:.2f}                             |   {res[("H", key)][1]:.2f}  {res[("V", key)][1]:.2f}')

    plt.rcParams.update({'font.size': 13})
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 4)
    kk = np.round(np.arange(0, 1.001, 0.1), 2)
    for r, o in enumerate(('H', 'V')):
        for c, lab in enumerate(('Divergence', 'Curl', 'Deformation')):
            a = fig.add_subplot(gs[r, c])
            for k, col in zip(kk, cm.jet(np.linspace(0, 1, len(kk)))):
                a.plot(P.phase, inv_unit(*tmpl_vel(k, o), o)[c], color=col, lw=1, alpha=.45)
            for key, name, col in runs[1:]:
                a.plot(P.phase, invs[(o, key)][c], 'o', color=col, ms=3, alpha=.7,
                       label=f'{name} (k = {res[(o, key)][0]:.2f})' if c == 1 else None)
            a.axhline(0, color='gray', lw=.8)
            a.axvspan(P.phase[0], P.phase[P.FIT][0], color='0.9'); a.axvspan(P.phase[P.FIT][-1], P.phase[-1], color='0.9')
            a.set(title=f'{"Horizontal" if o == "H" else "Vertical"} — {lab} (unit vectors)', xticks=[0, 45, 90, 135, 180])
            if c == 1:
                a.legend(fontsize=9)
        a = fig.add_subplot(gs[r, 3])
        for key, name, col in runs[1:]:
            a.plot(KG, sims[(o, key)], '-', color=col, lw=2.5, label=f'{name}: best k = {res[(o, key)][1]:.2f}')
            a.axvline(res[(o, key)][1], color=col, ls=':', lw=1.5)
        a.set(title=f'{"Horizontal" if o == "H" else "Vertical"} — direction matching', xlabel='template k', ylabel='mean cosine similarity')
        a.legend(fontsize=9)
    ax = fig.add_subplot(gs[2, :2])
    x = np.arange(len(runs))
    for i, (o, col) in enumerate((('H', 'r'), ('V', 'b'))):
        ax.bar(x+(i-.5)*.38, [res[(o, key)][0] for key, _, _ in runs], .38, color=col, label=f'{"Horizontal" if o == "H" else "Vertical"}')
    ax.set_xticks(x, [lab for _, lab, _ in runs]); ax.set(title='Method 2: best k (unit vectors → div, curl, def)', ylabel='k'); ax.legend()
    ax = fig.add_subplot(gs[2, 2:])
    for i, (o, col) in enumerate((('H', 'r'), ('V', 'b'))):
        ax.bar(x+(i-.5)*.38, [res[(o, key)][1] for key, _, _ in runs], .38, color=col, label=f'{"Horizontal" if o == "H" else "Vertical"}')
    ax.set_xticks(x, [lab for _, lab, _ in runs]); ax.set(title='Method 3: best k (direction matching)', ylabel='k'); ax.legend()
    fig.suptitle('Magnitude-free fits (thin lines: templates k = 0 blue … 1 red; dots: flows; grey: excluded frames)', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(P.A.out+'R2_37_direction_only_fit.png', dpi=80)
