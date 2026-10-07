#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Likelihood temperature sweep for the Gaussian Heeger flow of the rotating D4G ring (anisotropic x4, with the
isotropic cortex as reference). At each contour point the velocity is the mean of exp(-(cost - min)/tau),
tau = TEMP * range of the cost at that point; TEMP -> 0 is the cost minimum (Heeger's rule, grid argmin here).
Magnitude-free fits of k (unit-vector invariants, direction matching, atan2(def, curl)) and the def/curl residual.
New analysis; reads the existing code and data only.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_direction_only_fit as D
from Revision_R2_heeger_kfit_consistent import bank
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import sz_x, sz_y
from Revision_R2_D4G_ring import d4g_ring_video

P = D.P
out, data = P.A.out, P.A.data
TEMPS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.3]


def mean_flows(kind, orient):
    cache = data+f'heeger_temp_sweep_{kind}_{orient}.npz'
    if os.path.exists(cache):
        z = np.load(cache)
        return z['flows']
    t0 = time.time()
    video = d4g_ring_video(orient, sigma=1.5)
    L, _, _ = ring_points(orient)
    L = L[FRAMES]
    pts = (np.repeat(FRAMES, NP), np.clip(np.round(L[..., 0]), 0, sz_y-1).astype(int).ravel(),
           np.clip(np.round(L[..., 1]), 0, sz_x-1).astype(int).ravel())
    b = bank(kind)
    M = b.energies(video, pts)
    g, VX, VY = b.velocity_grid()
    vx, vy = VX.ravel(), VY.ravel()
    R = b.predicted(vx, vy)
    flows = np.zeros((len(TEMPS), len(M), 2))
    for s in range(0, len(M), 128):
        c = b.cost(M[s:s+128], R)
        c = c-c.min(1, keepdims=True)
        rng = np.ptp(c, 1, keepdims=True)
        for ti, T in enumerate(TEMPS):
            if T == 0:
                i = c.argmin(1)
                flows[ti, s:s+128] = np.stack([vx[i], vy[i]], 1)
            else:
                w = np.exp(-c/np.maximum(T*rng, 1e-12)); w /= w.sum(1, keepdims=True)
                flows[ti, s:s+128] = np.stack([w@vx, w@vy], 1)
    flows = flows.reshape(len(TEMPS), len(FRAMES), NP, 2)
    np.savez(cache, flows=flows)
    print(f'  {kind} {orient}: {(time.time()-t0)/60:.1f} min', flush=True)
    return flows


def fits(flow, orient):
    U, V = D.flow_phys(flow)
    inv = D.inv_unit(U, V, orient)
    a = np.degrees(np.arctan2(inv[2], inv[1]))
    TA = np.array([np.degrees(np.arctan2(t[2], t[1])) for t in D.T2[orient]])
    cost = np.sum((a[None, P.FIT]-TA[:, P.FIT])**2, 1)
    i = cost.argmin()
    rms = np.sqrt(cost[i]/P.FIT.stop) if isinstance(P.FIT, slice) else np.sqrt(cost[i]/len(a))
    return D.fit2(inv, orient), D.fit3(U, V, orient)[0], D.KG[i], np.sqrt(cost[i]/len(a[P.FIT])), a


if __name__ == '__main__':
    res, angs = {}, {}
    for kind in ('iso', 'aniso4'):
        for o in ('H', 'V'):
            F = mean_flows(kind, o)
            for ti, T in enumerate(TEMPS):
                k2, k3, ka, rms, a = fits(F[ti], o)
                res[(kind, o, T)] = (k2, k3, ka, rms); angs[(kind, o, T)] = a
    print('TEMP   | iso: unit / direction / def-curl (H=V)  | aniso ×4 H: unit / dir / def-curl (rms)  | aniso ×4 V: unit / dir / def-curl (rms)')
    for T in TEMPS:
        i, h, v = res[('iso', 'H', T)], res[('aniso4', 'H', T)], res[('aniso4', 'V', T)]
        lab = 'argmin' if T == 0 else f'{T:g}'
        print(f'{lab:6s} | {i[0]:.2f} / {i[1]:.2f} / {i[2]:.2f}                     | {h[0]:.2f} / {h[1]:.2f} / {h[2]:.2f} ({h[3]:4.1f}°)      | {v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f} ({v[3]:4.1f}°)')

    plt.rcParams.update({'font.size': 13})
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(2, 3)
    x = np.arange(len(TEMPS))
    xl = ['argmin' if T == 0 else f'{T:g}' for T in TEMPS]
    for c, (mi, name) in enumerate(((0, 'unit-vector invariants'), (1, 'direction matching'), (2, 'def/curl angle'))):
        a = fig.add_subplot(gs[0, c])
        a.plot(x, [res[('aniso4', 'H', T)][mi] for T in TEMPS], 'r-o', lw=2.5, label='anisotropic ×4, horizontal')
        a.plot(x, [res[('aniso4', 'V', T)][mi] for T in TEMPS], 'b-o', lw=2.5, label='anisotropic ×4, vertical')
        a.plot(x, [res[('iso', 'H', T)][mi] for T in TEMPS], 'k--o', lw=2, label='isotropic (H = V)')
        a.set(title=f'Best k: {name}', xlabel='temperature (fraction of cost range)', ylabel='k', ylim=[0, 1.05])
        a.set_xticks(x, xl); a.legend(fontsize=10)
    kk = np.round(np.arange(0, 1.001, 0.1), 2)
    showT = [0.0, 0.01, 0.05, 0.15]
    cols = cm.viridis(np.linspace(0, .9, len(showT)))
    for c, o in enumerate(('H', 'V')):
        a = fig.add_subplot(gs[1, c])
        for k, col in zip(kk, cm.jet(np.linspace(0, 1, len(kk)))):
            t = D.inv_unit(*D.tmpl_vel(k, o), o)
            a.plot(P.phase, np.degrees(np.arctan2(t[2], t[1])), color=col, lw=1, alpha=.35)
        for T, col in zip(showT, cols):
            r = res[('aniso4', o, T)]
            a.plot(P.phase, angs[('aniso4', o, T)], 'o', color=col, ms=3, alpha=.8,
                   label=f'{"argmin" if T == 0 else T}: k = {r[2]:.2f}, rms {r[3]:.1f}°')
        a.axhline(90, color='gray', ls='--', lw=.8)
        a.set(title=f'{"Horizontal" if o == "H" else "Vertical"}, anisotropic ×4: atan2(def, curl)', xlabel='Motion phase (deg)',
              xticks=[0, 45, 90, 135, 180], yticks=[0, 45, 90, 135, 180], ylim=[0, 180])
        a.legend(fontsize=10, loc='upper left')
    a = fig.add_subplot(gs[1, 2])
    a.plot(x, [res[('aniso4', 'H', T)][3] for T in TEMPS], 'r-o', lw=2.5, label='horizontal')
    a.plot(x, [res[('aniso4', 'V', T)][3] for T in TEMPS], 'b-o', lw=2.5, label='vertical')
    a.set(title='def/curl fit error (rms, deg), anisotropic ×4', xlabel='temperature', ylabel='deg'); a.set_xticks(x, xl); a.legend()
    fig.suptitle('Likelihood temperature sweep: mean of exp(-cost/τ) vs cost minimum (argmin)', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out+'R2_41_heeger_temperature_sweep.png', dpi=80)
