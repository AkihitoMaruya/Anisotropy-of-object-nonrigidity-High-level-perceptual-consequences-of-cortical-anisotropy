#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Width vs cell-number anisotropy (plain Heeger, full-width cat widths, folded, equal filter peaks, D4G ring):
  width sweep:  half-widths = mean x [1 + beta (h/mean - 1)], beta 0 ... 5 (step 0.5), EQUAL cell numbers
  number sweep: cell numbers = 1 + alpha (n_cat - 1) (folded cat numbers, mean 1), alpha 0 ... 5, EQUAL (mean) widths
Best k from direction matching (cosine similarity, as in Figure 5F/G), horizontal and vertical rotation.
New analysis; reads the existing code and data only.
"""
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
os.environ.setdefault('K_FIT_TRIM', '10')                        # frames 5-122 are decoded; dropping 10 more at each end
#   leaves video frames 15-112 (first and last 15 of the 128-frame cycle removed: occlusion near edge-on)
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from spherical_gabor_bank import exact_bank_sym, THETAS, fold, cat_hwhh
from anisotropy_heeger import cat_counts
import Revision_R2_heeger_temperature_sweep as TS
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import sz_x, sz_y
from Revision_R2_D4G_ring import d4g_ring_video
D, P = TS.D, TS.P
LEVELS = np.round(np.arange(0, 5.01, .5), 1)
SCALE, f0 = .5, .12
NCAT = fold(cat_counts(THETAS))


def bank(beta, alpha):
    b = exact_bank_sym(1)
    h = fold(cat_hwhh(beta))*SCALE
    s = np.repeat(2*f0*np.sin(h/2)/np.sqrt(2*np.log(2)), b.S)
    b.C = s[:, None, None]**2*np.eye(3)[None]
    b.w = np.repeat(1+alpha*(NCAT-1), b.S)
    return b


def kfit(fl, o):
    k3, sim = D.fit3(*D.flow_phys(fl), o)
    return k3, sim.max()


if __name__ == '__main__':
    res = {}
    for o in ('H', 'V'):
        cache = P.A.data+f'heeger_width_vs_number_{o}.npz'
        F = dict(np.load(cache)) if os.path.exists(cache) else {}
        prev = np.load(P.A.data+f'heeger_fwhh_equal_numbers_{o}.npz')
        video = None
        for kind, lv in [('w', l) for l in LEVELS]+[('n', l) for l in LEVELS]:
            key = f'{kind}{lv}'
            if key not in F:
                if kind == 'w' and f'b{lv:g}' in prev.files:
                    F[key] = prev[f'b{lv:g}']
                else:
                    t0 = time.time()
                    beta, alpha = (lv, 0) if kind == 'w' else (0, lv)
                    if kind == 'n':
                        M = np.load(P.A.data+f'heeger_fwhh_energies_x0_{o}.npz')['M']     # equal widths
                    else:
                        if video is None:
                            video = d4g_ring_video(o, sigma=1.5); Vf = np.fft.fftshift(np.fft.fftn(video))
                            L, _, _ = ring_points(o); L = L[FRAMES]
                            pts = (np.repeat(FRAMES, NP), np.clip(np.round(L[..., 0]), 0, sz_y-1).astype(int).ravel(),
                                   np.clip(np.round(L[..., 1]), 0, sz_x-1).astype(int).ravel())
                        M = bank(beta, 0).energies(video, pts, Vf)
                    F[key] = bank(beta, alpha).flow(M).reshape(len(FRAMES), NP, 2)
                    print(f'  {o} {key}: {(time.time()-t0)/60:.1f} min', flush=True)
                np.savez(cache, **F)
            res[(o, key)] = kfit(F[key], o)
    print(f'{"level":6s} | width only: k H / V (match) | numbers only: k H / V (match)')
    for lv in LEVELS:
        w = (res[('H', f'w{lv}')], res[('V', f'w{lv}')]); n = (res[('H', f'n{lv}')], res[('V', f'n{lv}')])
        print(f'{lv:<6g} | {w[0][0]:.2f} / {w[1][0]:.2f} ({w[0][1]:.2f} / {w[1][1]:.2f})      | {n[0][0]:.2f} / {n[1][0]:.2f} ({n[0][1]:.2f} / {n[1][1]:.2f})')

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 22, 'axes.linewidth': 2.5, 'xtick.major.width': 2.5,
                         'ytick.major.width': 2.5, 'xtick.major.size': 8, 'ytick.major.size': 8, 'axes.titlesize': 24,
                         'axes.labelsize': 24, 'legend.fontsize': 18})
    fig, ax = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    for a, kind, xl, title in ((ax[0], 'w', 'anisotropy of tuning width (× cat)', 'Width anisotropy, equal cell numbers'),
                               (ax[1], 'n', 'anisotropy of cell numbers (× cat)', 'Number anisotropy, equal widths')):
        a.plot(LEVELS, [res[('H', f'{kind}{l}')][0] for l in LEVELS], 'b-o', lw=5, ms=12, label='horizontal rotation')
        a.plot(LEVELS, [res[('V', f'{kind}{l}')][0] for l in LEVELS], 'r-o', lw=5, ms=12, label='vertical rotation')
        a.axvline(1, color='gray', ls=':', lw=3); a.text(1.08, 1.0, 'cat', color='gray', fontsize=20)
        a.set(xlabel=xl, title=title, xticks=LEVELS[::2], ylim=[0, 1.08]); a.legend(loc='lower left')
    ax[0].set_ylabel('Best-fitting k')
    plt.tight_layout(); plt.close(fig)   # the figure itself is made by Revision_R2_fig5_revised.py (R2_86_figS_anisotropy_strength.png)
