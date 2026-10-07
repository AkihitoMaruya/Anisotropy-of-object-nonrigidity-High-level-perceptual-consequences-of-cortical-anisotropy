#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised Figure 6F/G (the paper's 6E/F): best k against horizontal image stretch (1.0-1.5, 11 levels, as in
Figure6.py) for the isotropic and the anisotropic (width x2.5 cat) cortex, horizontal and vertical rotation, with the
Figure 5 model (one band f0 = 0.12, Heeger decoding, Horn-Schunck smoothing alpha = 1 x median speed) and the
deformation-over-curl fit of the revised Figure 6C/D (atan2(def, curl), least squares over all phases).
Stretch: each frame of the D4G ring video (sigma 1.5 px) is stretched horizontally about the image centre (linear
interpolation), and the contour points and template velocities with it (x -> s x, u -> s u). Invariants of a flow on a
closed contour by Green's theorem on the polygon (general for any outline): with tangent element t and outward normal
element n, A div = sum(u nx + v ny), A curl = sum(u tx + v ty), A def1 = sum(u nx - v ny), A def2 = sum(u ny + v nx).
Cache: heeger_fig6_stretch_{H,V}.npz, keys {U|A}_s{stretch}. Usage: python Revision_R2_fig6_stretch.py [minutes]
(time budget per run; run again to continue). New analysis.
"""
import os, sys, time
os.environ.setdefault('K_FIT_TRIM', '10')
import numpy as np
from scipy.ndimage import map_coordinates
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from spherical_gabor_bank import exact_bank_sym, THETAS, fold, cat_hwhh
from anisotropy_heeger import cat_counts
import Revision_R2_heeger_temperature_sweep as TS
import Revision_R2_fig5_hs_smooth as HS
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import sz_x, sz_y
from Revision_R2_D4G_ring import d4g_ring_video
import Revision_R2_heeger_kfit_consistent as C
D, P = TS.D, TS.P
STRETCHES = np.round(np.linspace(1, 1.5, 11), 2)
BETA, f0, SCALE, HS_ALPHA = 2.5, .12, .5, 1.0
NUM = fold(cat_counts(THETAS))
KG = np.round(np.arange(0, 1.5001, 0.01), 2)
CX = (sz_x-1)/2


def bank(kind):                                                    # as Revision_R2_fig5_revised.bank (ALPHA = 1)
    b = exact_bank_sym(1 if kind == 'A' else 0)
    h = fold(cat_hwhh(BETA if kind == 'A' else 0))*SCALE
    s = np.repeat(2*f0*np.sin(h/2)/np.sqrt(2*np.log(2)), b.S)
    b.C = s[:, None, None]**2*np.eye(3)[None]
    if kind == 'A':
        b.w = np.repeat(NUM, b.S)
    return b


def stretched_points(o, s):
    L = ring_points(o)[0][FRAMES].copy()
    L[..., 1] = CX+s*(L[..., 1]-CX)
    return L                                                         # (T, NP, 2) row, col


def stretched_video(video, s):
    if s == 1:
        return video
    rr, cc = np.meshgrid(np.arange(sz_y), CX+(np.arange(sz_x)-CX)/s, indexing='ij')
    return np.stack([map_coordinates(f, [rr, cc], order=1, mode='constant') for f in video])


def hs_pts(flow, L, alpha):
    """Horn-Schunck smoothing as Revision_R2_fig5_hs_smooth.hs, at given contour points L."""
    import torch
    rr = np.clip(np.round(L[..., 0]), 0, sz_y-1).astype(int); cc = np.clip(np.round(L[..., 1]), 0, sz_x-1).astype(int)
    T = len(L); tt = np.repeat(np.arange(T), NP)
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
    return np.stack([u[tt, rr.ravel(), cc.ravel()], v[tt, rr.ravel(), cc.ravel()]], 1).reshape(T, NP, 2)


def green_ratio(L, vel):
    """atan2(def, curl) (deg) per frame of velocity vel (T, NP, 2; screen x right, y up) on the closed contour L
    (row, col), by Green's theorem on the polygon."""
    x, y = L[..., 1], -L[..., 0]
    tx = (np.roll(x, -1, 1)-np.roll(x, 1, 1))/2; ty = (np.roll(y, -1, 1)-np.roll(y, 1, 1))/2
    A = 0.5*np.sum(x*np.roll(y, -1, 1)-np.roll(x, -1, 1)*y, 1)
    sg = np.sign(A)[:, None]; tx, ty = tx*sg, ty*sg                  # counter-clockwise
    nx, ny = ty, -tx
    u, v = vel[..., 0], vel[..., 1]
    curl = np.sum(u*tx+v*ty, 1); d1 = np.sum(u*nx-v*ny, 1); d2 = np.sum(u*ny+v*nx, 1)
    return np.degrees(np.arctan2(np.hypot(d1, d2), curl))           # area cancels


def template_ratios(o, s):
    L = stretched_points(o, s)
    out = []
    for k in KG:
        U, V = C.analytic_velocity(k, o)                              # ring coordinates; screen = -ring
        out.append(green_ratio(L, np.stack([-U*s, -V], -1)))
    return np.array(out)


def best_k(r, T):
    return KG[np.nansum((T-r[None])**2, 1).argmin()]


def flows(o, s, budget_end):
    cache = P.A.data+f'heeger_fig6_stretch_{o}.npz'; Z = dict(np.load(cache)) if os.path.exists(cache) else {}
    todo = [kind for kind in ('U', 'A') if f'{kind}_s{s}' not in Z]
    if todo and time.time() < budget_end:
        L = stretched_points(o, s)
        pts = (np.repeat(FRAMES, NP), np.clip(np.round(L[..., 0]), 0, sz_y-1).astype(int).ravel(),
               np.clip(np.round(L[..., 1]), 0, sz_x-1).astype(int).ravel())
        video = stretched_video(d4g_ring_video(o, sigma=1.5), s); Vf = np.fft.fftshift(np.fft.fftn(video))
        for kind in todo:
            t0 = time.time(); b = bank(kind)
            fl = b.flow(b.energies(video, pts, Vf)).reshape(len(FRAMES), NP, 2)
            Z[f'{kind}_s{s}'] = hs_pts(fl, L, HS_ALPHA*np.median(np.linalg.norm(fl, axis=-1))); np.savez(cache, **Z)
            print(f'  {o} {kind} stretch {s}: {(time.time()-t0)/60:.1f} min', flush=True)
    return Z


if __name__ == '__main__':
    end = time.time()+(float(sys.argv[1])*60 if len(sys.argv) > 1 else 1e9)
    for s in STRETCHES:
        for o in ('V', 'H'):
            flows(o, s, end)
    res = {}
    for o in ('H', 'V'):
        Z = dict(np.load(P.A.data+f'heeger_fig6_stretch_{o}.npz'))
        for s in STRETCHES:
            T = template_ratios(o, s); L = stretched_points(o, s)
            for kind in ('U', 'A'):
                if f'{kind}_s{s}' in Z:
                    res[(o, kind, s)] = best_k(green_ratio(L, Z[f'{kind}_s{s}']), T)
    for kind in ('U', 'A'):
        print(kind, 'V', [res.get(('V', kind, s)) for s in STRETCHES]); print(kind, 'H', [res.get(('H', kind, s)) for s in STRETCHES])
    np.save(P.A.data+'heeger_fig6_stretch_k.npy', res, allow_pickle=True)
    print('done' if len(res) == 4*len(STRETCHES) else 'incomplete: run again')
