#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model of Figure 5 (from Revision_R2_fig5_revised.py, plotting removed): Gaussian Heeger motion energy, spherical
filters with equal peaks, cat tuning widths read as full width at half height (Li et al. 2003), folded, width anisotropy
beta = 2.5, folded cat cell numbers; isotropic cortex: equal widths and numbers; one band f0 = 0.12 cycles/px; Heeger
decoding with the paper's Horn-Schunck smoothness constraint (alpha = 1 x median speed); D4G ring (sigma 1.5 px).
F[(o, kind)]: optic flow at the ring points (o: 'H'/'V'; kind: 'U' isotropic / 'A' anisotropic), cached in
Toolbox/Data/Rings/heeger_fig5_revised_{o}.npz and heeger_fig5_hs_{o}.npz (computed if missing).
COS[(kind, o)]: mean and SD of the cosine similarity between flow and template directions for each k (D.KG)."""
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
os.environ.setdefault('K_FIT_TRIM', '10')                        # frames 5-122 are decoded; dropping 10 more at each end
#   leaves video frames 15-112 (first and last 15 of the 128-frame cycle removed: occlusion near edge-on)
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from spherical_gabor_bank import exact_bank_sym, THETAS, LAYOUT_C, fold, cat_hwhh
from anisotropy_heeger import cat_counts
import Revision_R2_heeger_temperature_sweep as TS
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import Omega, sz_x, sz_y
from Revision_R2_D4G_ring import d4g_ring_video
D, P = TS.D, TS.P
BETA, f0, SCALE = 2.5, .12, .5
ALPHA = 1                                                        # cell-number anisotropy (x cat; 1 = cat numbers)
HS_ALPHA = 1.0                                                   # smoothness constraint weight (x median speed); 0 = none
THREE_BANDS = False                                              # three radial bands f0 = 0.065, 0.12, 0.22 (Revision_R2_three_band.py)
NUM = 1+ALPHA*(fold(cat_counts(THETAS))-1)                       # folded cat numbers (mean 1), anisotropy x ALPHA
out = P.A.out


def bank(kind):
    b = exact_bank_sym(1 if kind == 'A' else 0)                      # 'A': anisotropic numbers; 'U': equal numbers
    h = fold(cat_hwhh(BETA if kind == 'A' else 0))*SCALE
    s = np.repeat(2*f0*np.sin(h/2)/np.sqrt(2*np.log(2)), b.S)
    b.C = s[:, None, None]**2*np.eye(3)[None]
    if kind == 'A':
        b.w = np.repeat(NUM, b.S)
    return b


def dir_rgb(V):
    """Direction colour as in the original Figure 5E (Vis_vec_field: hue = atan2(-v, -u) / 2 pi, screen y up):
    leftward red, downward yellow-green, rightward cyan, upward purple."""
    hue = (np.arctan2(-V[..., 1], -V[..., 0])/(2*np.pi)) % 1
    return hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.full_like(hue, .9)], -1))


def direction_key(ax):
    ang = np.radians(np.arange(0, 360, 45)); v = np.stack([np.cos(ang), np.sin(ang)], 1)
    ax.quiver(np.zeros(8), np.zeros(8), v[:, 0], v[:, 1], color=dir_rgb(v), angles='xy', scale_units='xy', scale=1,
              width=.035, headwidth=3.5, headlength=4)
    for (x, y), t in zip(v[::2]*1.45, ('right', 'up', 'left', 'down')):
        ax.text(x, y, t, ha='center', va='center', fontsize=12)
    ax.set(xlim=[-1.9, 1.9], ylim=[-1.9, 1.9], aspect='equal'); ax.axis('off')


# flows
F = {}
for o in ('H', 'V'):
    cache = P.A.data+f'heeger_fig5_revised_{o}.npz'
    Z = dict(np.load(cache)) if os.path.exists(cache) else {}
    video = None
    for kind in ('U', 'A'):
        key = kind if (kind == 'U' or ALPHA == 1) else f'A_n{ALPHA}'      # 'A': cat numbers
        if key not in Z:
            t0 = time.time()
            if video is None:
                video = d4g_ring_video(o, sigma=1.5); Vf = np.fft.fftshift(np.fft.fftn(video))
                L, _, _ = ring_points(o); L = L[FRAMES]
                pts = (np.repeat(FRAMES, NP), np.clip(np.round(L[..., 0]), 0, sz_y-1).astype(int).ravel(),
                       np.clip(np.round(L[..., 1]), 0, sz_x-1).astype(int).ravel())
            b = bank(kind)
            Z[key] = b.flow(b.energies(video, pts, Vf)).reshape(len(FRAMES), NP, 2); np.savez(cache, **Z)
            print(f'  {o} {key}: {(time.time()-t0)/60:.1f} min', flush=True)
        F[(o, kind)] = Z[key]
    if HS_ALPHA:                                                   # the paper's smoothness constraint (Horn-Schunck)
        hz = P.A.data+f'heeger_fig5_hs_{o}.npz'
        hsz = dict(np.load(hz)) if os.path.exists(hz) else {}
        for kind in ('U', 'A'):
            key = f'{kind}_a{HS_ALPHA}'
            if key not in hsz:                                       # Horn-Schunck smoothing (Revision_R2_fig5_hs_smooth.py)
                import Revision_R2_fig5_hs_smooth as HS
                hsz[key] = HS.hs(F[(o, kind)], o, HS_ALPHA*np.median(np.linalg.norm(F[(o, kind)], axis=-1))); np.savez(hz, **hsz)
            F[(o, kind)] = hsz[key]


COS = {}
for kind in ('U', 'A'):
    for o in ('H', 'V'):
        U_, V_ = D.flow_phys(F[(o, kind)]); u, v = D.unit(U_, V_); ok = np.isfinite(u) & (np.hypot(U_, V_) > 1e-9)
        cs = [(u*T[0]+v*T[1])[P.FIT][ok[P.FIT]] for T in D.TU[o]]
        COS[(kind, o)] = (np.array([c.mean() for c in cs]), np.array([c.std() for c in cs]))
K = D.KG
LP = {o: ring_points(o)[0][FRAMES] for o in ('H', 'V')}
FI45 = int(np.argmin(np.abs(np.degrees(Omega[FRAMES])-45)))
