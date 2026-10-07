#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotation (k = 0) vs wobble (k = 1) fit of the Gaussian Heeger flows of the rotating D4G ring, with the
procedure of Figure 5C / 6: per frame, integrate the flow against the div, curl and def operator
fields along the ring, normalise the three curves by their joint maximum, and fit k by least squares
to the analytic templates (Figure6.py). The first and last TRIM frames (ring near edge-on) are dropped
before normalising and fitting. Screen flows (x right, y up, px/frame) are converted to the
ring's physical frame: v_phys = -v_screen (the rendering flips both axes); scale drops out.
Checks: the true rigid-rotation velocity (expected k ~ 0) and the normal flow.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import TILT, Omega as OmegaAll

out = current_folder+'/Toolbox/Data/work/'
data = current_folder+'/Toolbox/Data/Rings/'
phi = TILT['bottom']
tt = np.linspace(0, 2*np.pi, NP+1)[:-1]                      # the ring points of the flows
Om = OmegaAll[FRAMES]
TRIM = int(os.environ.get('K_FIT_TRIM', 4))                 # frames dropped at each end (near edge-on)
KEEP = slice(TRIM, len(Om)-TRIM)
FLIP_V_CURL = os.environ.get('K_FIT_FLIP_V_CURL', '0') == '1'   # test: reverse the curl sign for vertical rotation
Om_fit = Om[KEEP]


def rotate(x, y, rot=np.pi/2):
    return np.cos(rot)*x-np.sin(rot)*y, np.sin(rot)*x+np.cos(rot)*y


def fields(orient, th=tt):
    """Figure5C curl / div / def operator fields at ring angles th, frames FRAMES."""
    curl1 = -(-np.outer(np.cos(Om)*np.cos(phi), np.sin(th))+np.outer(np.sin(Om), np.cos(th)))
    curl2 = -(-np.outer(np.sin(phi)*np.ones_like(Om), np.sin(th)))
    div1 = np.outer(np.sin(phi)*np.ones_like(Om), np.sin(th))
    div2 = -np.outer(np.cos(Om)*np.cos(phi), np.sin(th))+np.outer(np.sin(Om), np.cos(th))
    if orient == 'V':
        curl1, curl2 = rotate(curl1, curl2)
        div1, div2 = rotate(div1, div2)
    return dict(div=(div1, div2), curl=(curl1, curl2), def1=(-curl1, curl2), def2=(-div1, div2))


def invariants(flow_screen, orient):
    U, V = -flow_screen[..., 0], -flow_screen[..., 1]            # physical frame
    F = fields(orient)
    I = {g: np.sum(f[0]*U+f[1]*V, axis=1) for g, f in F.items()}
    div, curl, dfm = I['div'][KEEP], I['curl'][KEEP], np.hypot(I['def1'], I['def2'])[KEEP]
    if FLIP_V_CURL and orient == 'V':
        curl = -curl
    n = max(np.abs(div).max(), np.abs(curl).max(), np.abs(dfm).max())
    return div/n, curl/n, dfm/n


# templates (Figure6.py, as in Revision_R2_def_curl.template), same frames
theta500 = np.linspace(0, 2*np.pi, 500)


def template(k):
    O, th = Om_fit, theta500
    curlw1 = -(-np.outer(np.cos(O)*np.cos(k*O)+np.sin(O)*np.sin(k*O), np.sin(th))+np.outer(np.sin(O)*np.cos(k*O)-np.sin(k*O)*np.cos(O)*np.cos(phi), np.cos(th)))
    curlw2 = -(np.sin(phi)*(-np.outer(np.cos(k*O), np.sin(th))-np.outer(np.sin(k*O), np.cos(th))))
    vec1w = (np.outer(-np.sin(O)*np.cos(k*O)*np.cos(phi)-k*np.cos(O)*np.sin(k*O)*np.cos(phi)+np.cos(O)*np.sin(k*O)+k*np.sin(O)*np.cos(k*O), np.cos(th))
             + np.outer(np.cos(O)*np.cos(k*O)-k*np.sin(O)*np.sin(k*O)-k*np.cos(k*O)*np.cos(O)*np.cos(phi)+np.sin(k*O)*np.sin(O)*np.cos(phi), np.sin(th)))
    vec2w = -np.sin(phi)*(np.outer(np.sin(k*O), np.cos(th))+np.outer(np.cos(k*O), np.sin(th)))*k
    divw1, divw2 = curlw2, -curlw1
    d = np.sum(divw1*vec1w+divw2*vec2w, axis=1)
    c = np.sum(curlw1*vec1w+curlw2*vec2w, axis=1)
    f = np.hypot(np.sum(curlw1*vec1w-curlw2*vec2w, axis=1), np.sum(divw1*vec1w-divw2*vec2w, axis=1))
    n = max(np.abs(d).max(), np.abs(c).max(), np.abs(f).max())
    return d/n, c/n, f/n


KG = np.linspace(0, 1.2, 121)
TPL = [template(k) for k in KG]


def best_k(inv):
    cost = np.array([sum(np.sum((a-b)**2) for a, b in zip(inv, T)) for T in TPL])
    k0 = KG[cost.argmin()]
    r = minimize_scalar(lambda k: sum(np.sum((a-b)**2) for a, b in zip(inv, template(k))),
                        bounds=(max(0, k0-.02), min(1.2, k0+.02)), method='bounded')
    return r.x, r.fun


if __name__ == '__main__':
    runs = [('cat ×1', '', 'Anisotropic'), ('cat ×2', '_beta2', 'Anisotropic'), ('cat ×3', '_beta3', 'Anisotropic'),
            ('cat ×3, pooled', '_beta3_pool4', 'Anisotropic'), ('cat ×4', '_beta4', 'Anisotropic'), ('cat ×4, pooled', '_beta4_pool4', 'Anisotropic')]
    res = {}
    for orient in ('H', 'V'):
        L, v_true, n_scr = ring_points(orient)
        vt = v_true[FRAMES]
        vn = np.sum(vt*n_scr[FRAMES], -1, keepdims=True)*n_scr[FRAMES]
        res[(orient, 'true velocity')] = best_k(invariants(vt, orient))
        res[(orient, 'normal flow')] = best_k(invariants(vn, orient))
        d0 = np.load(data+f'heeger_layoutC_{orient}.npz')
        res[(orient, 'isotropic')] = best_k(invariants(d0['iso'], orient))
        dp = np.load(data+f'heeger_layoutC_beta3_pool4_{orient}.npz')
        res[(orient, 'isotropic, pooled')] = best_k(invariants(dp['iso'], orient))
        for name, suf, key in runs:
            d = np.load(data+f'heeger_layoutC{suf}_{orient}.npz')
            res[(orient, name)] = best_k(invariants(d['aniso'], orient))
    names = ['true velocity', 'normal flow', 'isotropic', 'isotropic, pooled']+[r[0] for r in runs]
    for nm in names:
        print(f'{nm:18s}: k  H {res[("H", nm)][0]:.2f}   V {res[("V", nm)][0]:.2f}   (residual H {res[("H", nm)][1]:.2f}, V {res[("V", nm)][1]:.2f})')
    np.save(data+('heeger_k_fit_flipVcurl.npy' if FLIP_V_CURL else 'heeger_k_fit.npy'), res, allow_pickle=True)

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    ax.bar(x-.2, [res[('H', n)][0] for n in names], .4, color='r', label='Horizontal rotation')
    ax.bar(x+.2, [res[('V', n)][0] for n in names], .4, color='b', label='Vertical rotation')
    ax.set_xticks(x, [n.replace(', ', ',\n') for n in names], fontsize=12)
    ax.set(ylabel='Best-fit k (0 rotation, 1 wobble)', ylim=[0, 1.1],
           title='Rotation vs wobble fit of the Heeger flows (layout C, raw normalisation, D4G ring σ 1.5 px, f₀ 0.12)')
    ax.axvline(1.5, color='gray', ls=':')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out+('R2_23_heeger_k_fit_flipVcurl.png' if FLIP_V_CURL else 'R2_23_heeger_k_fit.png'), dpi=100)
