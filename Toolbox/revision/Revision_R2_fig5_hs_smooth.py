#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The paper's smoothness constraint (Heeger_pyr_flow(smoothness=True), Toolbox/Compute_optic_flow_from_3D_gabor.py)
applied to the Figure 5 flows (Revision_R2_fig5_revised.py): Horn-Schunck iteration on the image grid, frame by frame,
with the local estimate v_hat as the constraint fx = u_hat, fy = v_hat, ft = -|v_hat|^2 (component along v_hat fixed),
3x3 averaging kernel (1/12, 1/6), u <- u_avg - fx p/d, p = fx u_avg + fy v_avg + ft, d = 4 alpha^2 + fx^2 + fy^2,
stop when the change < 1e-2 or after 1000 iterations (scipy 'reflect' boundary as in ndimage.convolve). alpha is given relative to the median estimated speed (the
paper's alpha = 100 refers to its own velocity units). Best k (direction matching) and match quality.
New analysis; reads the existing code and data only.
"""
import os, sys, time
import numpy as np
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_heeger_temperature_sweep as TS
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import sz_x, sz_y
D, P = TS.D, TS.P
ALPHAS = [0.1, 0.3, 1.0]
KERNEL = np.array([[1/12, 1/6, 1/12], [1/6, 0, 1/6], [1/12, 1/6, 1/12]])


def hs(flow, o, alpha):
    """flow (T, NP, 2) screen (x right, y up) -> smoothed flow at the same contour points. All frames at once (torch);
    per-frame iteration as in the paper's code, each frame stopped when its change < 1e-2 (at most 1000 iterations)."""
    import torch
    L = ring_points(o)[0][FRAMES]
    rr = np.clip(np.round(L[..., 0]), 0, sz_y-1).astype(int); cc = np.clip(np.round(L[..., 1]), 0, sz_x-1).astype(int)
    T = len(FRAMES); tt = np.repeat(np.arange(T), NP)
    fx = np.zeros((T, sz_y, sz_x)); fy = np.zeros((T, sz_y, sz_x))
    fx[tt, rr.ravel(), cc.ravel()] = flow[..., 0].ravel(); fy[tt, rr.ravel(), cc.ravel()] = flow[..., 1].ravel()
    fx, fy = torch.tensor(fx), torch.tensor(fy)
    ft = -(fx**2+fy**2); d = 4*alpha**2+fx**2+fy**2
    k = torch.tensor(KERNEL)[None, None]
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


if __name__ == '__main__':
    res = {}
    for o in ('H', 'V'):
        cache = P.A.data+f'heeger_fig5_hs_{o}.npz'
        F = dict(np.load(cache)) if os.path.exists(cache) else {}
        fig5 = np.load(P.A.data+f'heeger_fig5_revised_{o}.npz')
        for kind in ('U', 'A'):
            fl = fig5[kind]; sp = np.median(np.linalg.norm(fl, axis=-1))
            k3, sim = D.fit3(*D.flow_phys(fl), o); res[(o, kind, 0)] = (k3, sim.max())
            for a in ALPHAS:
                key = f'{kind}_a{a}'
                if key not in F:
                    t0 = time.time(); F[key] = hs(fl, o, a*sp); np.savez(cache, **F)
                    print(f'  {o} {key}: {(time.time()-t0)/60:.1f} min', flush=True)
                k3, sim = D.fit3(*D.flow_phys(F[key]), o); res[(o, kind, a)] = (k3, sim.max())
    print(f'{"alpha (x median speed)":22s} | isotropic H / V (match)   | anisotropic H / V (match)')
    for a in [0]+ALPHAS:
        u = (res[('H', 'U', a)], res[('V', 'U', a)]); an = (res[('H', 'A', a)], res[('V', 'A', a)])
        print(f'{("none" if a == 0 else a):<22} | {u[0][0]:.2f} / {u[1][0]:.2f} ({u[0][1]:.2f} / {u[1][1]:.2f})   | {an[0][0]:.2f} / {an[1][0]:.2f} ({an[0][1]:.2f} / {an[1][1]:.2f})')
