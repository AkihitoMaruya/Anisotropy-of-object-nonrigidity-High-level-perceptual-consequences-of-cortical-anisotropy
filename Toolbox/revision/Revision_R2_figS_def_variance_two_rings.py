#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary: variance of local deformation for the joined two rings. Stimulus: the model's D4G rings (sigma 1.5 px)
with orthographic projection (Revision_R2_two_rings_ME.stimulus_and_fields): the bottom ring (tilt +30 deg) shifted
down by sin 30 and the top ring (tilt -30 deg) shifted up by sin 30, so that they touch at the joint, in one video;
vertical rotation = the image rotated by 90 deg. Optic flow of the Figure 5 model (one band, Heeger decoding,
Horn-Schunck smoothing alpha = 1 x median speed) on the centre lines of both rings, isotropic and anisotropic
(width x2.5 cat) cortex. Local deformation from triplets of contour points over both rings (three velocities ->
exact affine flow; def = sqrt((du/dx - dv/dy)^2 + (du/dy + dv/dx)^2), 1/frame); N_TRI random triplets per frame,
kept if the triangle area >= MIN_AREA x the area enclosed by one ring. Variance across triplets at each phase.
References: true velocity of rigid rotation (k = 0) and of wobble (k = 1).
Cache heeger_two_ring_def_{o}.npz. Output: Images/Revision_R2/R2_86_figS_def_variance_two_rings.png. New analysis.
"""
import os, sys, time
os.environ.setdefault('K_FIT_TRIM', '10')
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_fig6_stretch as G                                # bank (Figure 5 model)
import Revision_R2_fig5_hs_smooth as HS
from Revision_R2_heeger_gaussian_flow import FRAMES, NP
from Revision_R2_two_rings_ME import stimulus_and_fields, TILT, Omega, sz_x, sz_y, rotate
from Revision_R2_D4G_fit import d4g, centre_line_pixels, pix
P = G.P
plt.rcParams.update({'font.family': 'Arial', 'font.size': 22, 'axes.linewidth': 1.8})
N_TRI, MIN_AREA = 3000, 0.1
COL = {'V': '#c0392b', 'H': '#2471a3'}
phase = np.degrees(Omega[FRAMES])
SHIFT = {'bottom': -np.sin(TILT['bottom']), 'top': np.sin(TILT['bottom'])}


def ring_pos(o, ring):
    pos1, pos2, _, _, _ = stimulus_and_fields(TILT[ring], 'H')
    pos2 = pos2+SHIFT[ring]
    if o == 'V':
        pos1, pos2 = rotate(pos1, pos2)
    return pos1, pos2


def two_ring_points(o):
    """Centre-line points (row, col) of both rings (NP each), all frames, and their true velocity (screen px/frame)."""
    L = np.concatenate([np.array([centre_line_pixels(p1[t], p2[t], n=NP+1)[:-1] for t in range(len(Omega))])
                        for p1, p2 in [ring_pos(o, r) for r in ('bottom', 'top')]], 1)
    vel = np.zeros_like(L); vel[1:-1] = (L[2:]-L[:-2])/2
    return L, np.stack([vel[..., 1], -vel[..., 0]], -1)


def two_ring_points_k(o, k):
    """As two_ring_points for motion mix k (k = 1: wobble): same outlines and the same fixed contour positions (ring
    parameter theta_i, so the joint stays at index 0); the velocity there is that of the material point currently at
    theta_i, which under mix k sits at parameter theta_mat + k Omega (X_k = Ry(Omega) R_n(-k Omega) p;
    Revision_R2_two_rings.ring_motion): v = (P(theta_i + k dOmega, t+1) - P(theta_i - k dOmega, t-1)) / 2."""
    th = np.linspace(0, 2*np.pi, NP+1)[:-1]; tt = np.linspace(0, 2*np.pi, 4001); dO = Omega[1]-Omega[0]
    Ls, Vs = [], []
    for p1, p2 in [ring_pos(o, r) for r in ('bottom', 'top')]:
        def at(t, q):
            dense = centre_line_pixels(p1[t], p2[t], n=len(tt)); q = q % (2*np.pi)
            return np.stack([np.interp(q, tt, dense[:, 0]), np.interp(q, tt, dense[:, 1])], 1)
        L = np.array([at(t, th) for t in range(len(Omega))])
        vel = np.zeros_like(L)
        for t in range(1, len(Omega)-1):
            vel[t] = (at(t+1, th+k*dO)-at(t-1, th-k*dO))/2
        Ls.append(L); Vs.append(vel)
    L, vel = np.concatenate(Ls, 1), np.concatenate(Vs, 1)
    return L, np.stack([vel[..., 1], -vel[..., 0]], -1)


def two_ring_video(o, sigma=1.5):
    rings = [ring_pos(o, r) for r in ('bottom', 'top')]
    v = np.zeros((len(Omega), sz_y, sz_x))
    for t in range(len(Omega)):
        line = np.concatenate([centre_line_pixels(p1[t], p2[t]) for p1, p2 in rings])
        v[t] = d4g(cKDTree(line).query(pix)[0].reshape(sz_y, sz_x), sigma)/3
    return v


def hs_pts(flow, L, alpha):
    """Horn-Schunck smoothing (Revision_R2_fig5_hs_smooth.hs) at any set of points L (T, n, 2)."""
    import torch
    T, n = L.shape[:2]
    rr = np.clip(np.round(L[..., 0]), 0, sz_y-1).astype(int); cc = np.clip(np.round(L[..., 1]), 0, sz_x-1).astype(int)
    tt = np.repeat(np.arange(T), n)
    fx = np.zeros((T, sz_y, sz_x)); fy = np.zeros((T, sz_y, sz_x))
    fx[tt, rr.ravel(), cc.ravel()] = flow[..., 0].ravel(); fy[tt, rr.ravel(), cc.ravel()] = flow[..., 1].ravel()
    fx, fy = torch.tensor(fx), torch.tensor(fy)
    ft = -(fx**2+fy**2); d = 4*alpha**2+fx**2+fy**2
    k = torch.tensor(HS.KERNEL)[None, None]
    u = torch.zeros_like(fx); v = torch.zeros_like(fy); active = torch.ones(T, dtype=torch.bool)
    conv = lambda x: torch.nn.functional.conv2d(torch.nn.functional.pad(x[:, None], (1, 1, 1, 1), mode='reflect'), k)[:, 0]
    for it in range(1000):
        ua, va = conv(u), conv(v)
        p = fx*ua+fy*va+ft
        un = ua-fx*p/d; vn = va-fy*p/d
        diff = torch.linalg.norm((un-u).reshape(T, -1), dim=1)
        u = torch.where(active[:, None, None], un, u); v = torch.where(active[:, None, None], vn, v)
        active &= diff >= 1e-2
        if not active.any():
            break
    u, v = u.numpy(), v.numpy()
    return np.stack([u[tt, rr.ravel(), cc.ravel()], v[tt, rr.ravel(), cc.ravel()]], 1).reshape(T, n, 2)


def flows(o):
    cache = P.A.data+f'heeger_two_ring_def_{o}.npz'; Z = dict(np.load(cache)) if os.path.exists(cache) else {}
    if not all(k in Z for k in ('U', 'A')):
        L = two_ring_points(o)[0][FRAMES]; n = L.shape[1]
        pts = (np.repeat(FRAMES, n), np.clip(np.round(L[..., 0]), 0, sz_y-1).astype(int).ravel(),
               np.clip(np.round(L[..., 1]), 0, sz_x-1).astype(int).ravel())
        video = two_ring_video(o); Vf = np.fft.fftshift(np.fft.fftn(video))
        for kind in ('U', 'A'):
            if kind in Z:
                continue
            t0 = time.time(); b = G.bank(kind)
            fl = b.flow(b.energies(video, pts, Vf)).reshape(len(FRAMES), n, 2)
            Z[kind] = hs_pts(fl, L, G.HS_ALPHA*np.median(np.linalg.norm(fl, axis=-1))); np.savez(cache, **Z)
            print(f'  two rings {o} {kind}: {(time.time()-t0)/60:.1f} min', flush=True)
    return Z


def triplet_def(L, vel, seed=0):
    rng = np.random.default_rng(seed); out = []
    for Lt, vt in zip(L, vel):
        x, y = Lt[:, 1], -Lt[:, 0]
        xb, yb = x[:NP], y[:NP]
        A = 0.5*abs(np.sum(xb*np.roll(yb, -1)-np.roll(xb, -1)*yb))       # area of one ring
        idx = rng.integers(0, len(x), (6*N_TRI, 3))
        X, Y = x[idx], y[idx]
        tri = 0.5*np.abs((X[:, 1]-X[:, 0])*(Y[:, 2]-Y[:, 0])-(X[:, 2]-X[:, 0])*(Y[:, 1]-Y[:, 0]))
        idx = idx[tri >= MIN_AREA*A][:N_TRI]
        M = np.stack([np.ones(idx.shape), x[idx], y[idx]], -1)
        a = np.linalg.solve(M, vt[idx, 0][..., None])[..., 0]; b = np.linalg.solve(M, vt[idx, 1][..., None])[..., 0]
        out.append(np.hypot(a[:, 1]-b[:, 2], a[:, 2]+b[:, 1]))
    return out


if __name__ == '__main__':
    VAR = {}
    for o in ('V', 'H'):
        Z = flows(o); L, vtrue = two_ring_points(o); L, vtrue = L[FRAMES], vtrue[FRAMES]
        VAR[(o, 'true')] = np.array([np.var(d) for d in triplet_def(L, vtrue)])
        Lw, vw = two_ring_points_k(o, 1.0)
        VAR[(o, 'wobble')] = np.array([np.var(d) for d in triplet_def(Lw[FRAMES], vw[FRAMES])])
        for kind in ('U', 'A'):
            VAR[(o, kind)] = np.array([np.var(d) for d in triplet_def(L, Z[kind])])
    fig, ax = plt.subplots(figsize=(13, 8))
    for o in ('V', 'H'):
        lab = 'vertical' if o == 'V' else 'horizontal'
        for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
            ax.plot(phase, VAR[(o, kind)], ls, color=COL[o], lw=3.5, alpha=.55 if kind == 'U' else 1, label=f'{lab} rotation, {cl}')
    ax.plot(phase, VAR[('V', 'true')], '--', color='0.5', lw=2.5, label='rigid rotation (true velocity)')
    ax.plot(phase, VAR[('V', 'wobble')], '--', color='k', lw=2.5, label='wobble (true velocity)')
    ax.set(xlabel='Motion phase (deg)', ylabel='Variance of local deformation (frame$^{-2}$)', xticks=[0, 45, 90, 135, 180],
           xlim=[0, 180], yscale='log')
    ax.set_title('Two rings', loc='left', fontweight='bold')
    ax.legend(fontsize=15, frameon=False, loc='upper center')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(P.A.out+'R2_86_figS_def_variance_two_rings.png', dpi=80)
    mid = np.abs(phase-90) < 10
    for key, v in VAR.items():
        print(key, f'near 90 deg {v[mid].mean():.3g}; median {np.median(v):.3g}')
