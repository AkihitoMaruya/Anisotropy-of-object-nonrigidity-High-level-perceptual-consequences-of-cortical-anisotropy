#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heeger optic flow of the rotating D4G ring from the derived 3D Gaussian filter bank
(Motion_texture/heeger_gaussian_flow.py via Toolbox/anisotropy_heeger.py), for horizontal and vertical rotation, isotropic and
anisotropic cortex. Saves the flows and plots them with direction colour-coded (hue = direction
on the screen, as in the direction wheel), next to the true velocity and the normal flow.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import imageio.v2 as imageio

current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from anisotropy_heeger import GaussianBank, cat_counts
from gaussian_angle_masks3D import cat_sigma
from Revision_R2_two_rings_ME import stimulus_and_fields, TILT, sz_x, sz_y, Omega
from Revision_R2_D4G_fit import centre_line_pixels
from Revision_R2_D4G_ring import d4g_ring_video

out = current_folder+'/Toolbox/Data/work/'
data_out = current_folder+'/Toolbox/Data/Rings/'
os.makedirs(data_out, exist_ok=True)

K = 16
thetas = np.arange(K)*2*np.pi/K
phis = np.radians([-45, 0, 45])
BANKS = {'iso': GaussianBank(thetas, np.full(K, cat_sigma(thetas).mean()), phis),
         'aniso': GaussianBank(thetas, cat_sigma(thetas), phis, weights=cat_counts(thetas))}
NP = 240                                                    # points along the ring
FRAMES = np.arange(5, len(Omega)-5)


def ring_points(orient):
    """Centre-line points (row, col) per frame, true velocity and unit normal in screen coords
    (x right, y up), px/frame."""
    pos1, pos2, _, _, _ = stimulus_and_fields(TILT['bottom'], orient)
    L = np.array([centre_line_pixels(pos1[t], pos2[t], n=NP+1)[:-1] for t in range(len(Omega))])  # (T, NP, 2)
    vel = np.zeros_like(L)
    vel[1:-1] = (L[2:]-L[:-2])/2
    v_scr = np.stack([vel[..., 1], -vel[..., 0]], -1)
    tan = np.roll(L, -1, 1)-np.roll(L, 1, 1)
    t_scr = np.stack([tan[..., 1], -tan[..., 0]], -1)
    n_scr = np.stack([t_scr[..., 1], -t_scr[..., 0]], -1)
    n_scr /= np.linalg.norm(n_scr, axis=-1, keepdims=True)
    return L, v_scr, n_scr


def direction_rgb(v, vmax):
    ang = np.arctan2(v[..., 1], v[..., 0])
    sp = np.linalg.norm(v, axis=-1)
    return hsv_to_rgb(np.stack([(ang % (2*np.pi))/(2*np.pi), np.ones_like(ang), np.clip(0.35+0.65*sp/vmax, 0, 1)], -1))


if __name__ == '__main__':
    res = {}
    for orient in ('H', 'V'):
        video = d4g_ring_video(orient)
        L, v_true, n_scr = ring_points(orient)
        rows = np.clip(np.round(L[FRAMES, :, 0]), 0, sz_y-1).astype(int).ravel()
        cols = np.clip(np.round(L[FRAMES, :, 1]), 0, sz_x-1).astype(int).ravel()
        tt = np.repeat(FRAMES, NP)
        for b, bank in BANKS.items():
            t0 = time.time()
            M = bank.energies(video, (tt, rows, cols))
            est = bank.flow(M).reshape(len(FRAMES), NP, 2)
            res[(orient, b)] = est
            np.savez(data_out+f'heeger_gauss_{orient}_{b}.npz', flow=est, frames=FRAMES, points=L[FRAMES],
                     v_true=v_true[FRAMES], normal=n_scr[FRAMES])
            print(f'{orient} {b}: {(time.time()-t0)/60:.1f} min', flush=True)
        res[(orient, 'video')] = video
        res[(orient, 'L')] = L[FRAMES]
        res[(orient, 'true')] = v_true[FRAMES]
        vn = np.sum(v_true*n_scr, -1, keepdims=True)*n_scr
        res[(orient, 'normal')] = vn[FRAMES]

    # summary: angle between estimated flow and the normal flow (+ = counter-clockwise)
    for orient in ('H', 'V'):
        for b in ('iso', 'aniso'):
            e, nf = res[(orient, b)], res[(orient, 'normal')]
            ok = (np.linalg.norm(nf, axis=-1) > .2) & (np.linalg.norm(e, axis=-1) > .05)
            d = np.degrees(np.angle(np.exp(1j*(np.arctan2(e[..., 1], e[..., 0])-np.arctan2(nf[..., 1], nf[..., 0])))))
            hz = np.abs(e[..., 0])/np.maximum(np.linalg.norm(e, axis=-1), 1e-9)
            hz_n = np.abs(nf[..., 0])/np.maximum(np.linalg.norm(nf, axis=-1), 1e-9)
            print(f'{orient} {b}: |angle from normal| median {np.median(np.abs(d[ok])):.1f} deg; '
                  f'horizontal share |cos| flow {hz[ok].mean():.3f} vs normal {hz_n[ok].mean():.3f}')

    # figure: two frames, rows H / V, columns true | normal | iso | aniso
    vmax = 2.5
    show = [int(np.argmin(np.abs(np.degrees(Omega[FRAMES])-a))) for a in (45, 90)]
    cols_ = [('true', 'True velocity'), ('normal', 'Normal flow'), ('iso', 'Heeger, isotropic'), ('aniso', 'Heeger, anisotropic')]
    fig, ax = plt.subplots(4, 4, figsize=(20, 20))
    for r, (orient, fi) in enumerate([(o, f) for o in ('H', 'V') for f in show]):
        t = FRAMES[fi]
        for c, (key, ttl) in enumerate(cols_):
            a = ax[r, c]
            a.imshow(res[(orient, 'video')][t], cmap='gray', vmin=-1.5, vmax=1.5, alpha=.5)
            P, V = res[(orient, 'L')][fi], res[(orient, key)][fi]
            s = slice(None, None, 4)
            a.quiver(P[s, 1], P[s, 0], V[s, 0], -V[s, 1], color=direction_rgb(V[s], vmax), angles='xy',
                     scale_units='xy', scale=.12, width=.006, headwidth=3.5)
            a.set_title(f'{"Horizontal" if orient == "H" else "Vertical"} rot., {np.degrees(Omega[t]):.0f}° — {ttl}', fontsize=13)
            a.axis('off')
    wax = fig.add_axes([.93, .45, .06, .06])
    g = np.linspace(-1, 1, 101); GX, GY = np.meshgrid(g, -g)
    wheel = direction_rgb(np.stack([GX, GY], -1)*vmax, vmax); wheel[np.hypot(GX, GY) > 1] = 1
    wax.imshow(wheel); wax.axis('off'); wax.set_title('direction', fontsize=11)
    plt.tight_layout(rect=[0, 0, .92, 1])
    fig.savefig(out+'R2_8_heeger_gauss_flow.png', dpi=90)

    # video: iso / aniso for H and V
    frames = []
    for fi, t in enumerate(FRAMES):
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        for r, orient in enumerate(('H', 'V')):
            for c, key in enumerate(('normal', 'iso', 'aniso')):
                a = ax[r, c]
                a.imshow(res[(orient, 'video')][t], cmap='gray', vmin=-1.5, vmax=1.5, alpha=.5)
                P, V = res[(orient, 'L')][fi], res[(orient, key)][fi]
                s = slice(None, None, 4)
                a.quiver(P[s, 1], P[s, 0], V[s, 0], -V[s, 1], color=direction_rgb(V[s], vmax), angles='xy',
                         scale_units='xy', scale=.12, width=.006, headwidth=3.5)
                a.set_title(f'{"Horizontal" if orient == "H" else "Vertical"} — {dict(cols_)[key]}', fontsize=13)
                a.axis('off')
        fig.suptitle(f'Phase {np.degrees(Omega[t]):.0f}°')
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out+'R2_8_heeger_gauss_flow.mp4', frames, fps=12)
