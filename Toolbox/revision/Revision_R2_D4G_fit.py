#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Approximate each frame of the rendered ring (make_rotating_stim, as fed to the ME model)
by a contour with a 4th-derivative-of-Gaussian (D4G) cross-section:

    I(x) ~ A * D4G_sigma(d(x)),   d(x) = distance of pixel x from the ring's centre line
    D4G_sigma(d) = (d^4/sigma^4 - 6 d^2/sigma^2 + 3) exp(-d^2 / (2 sigma^2))   (peak 3 at d=0)

For each frame, A is solved by least squares (background is 0, so no offset) on a grid of sigma, and the best sigma
is kept. The 1D spectrum of this profile is |F(rho)| ~ rho^4 exp(-sigma^2 rho^2 / 2): a single
band-pass peak at rho = 2/sigma rad/pixel (1/(pi sigma) cycles/pixel), no DC, no side lobes.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import imageio.v2 as imageio

current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from Vis_vec_field import make_rotating_stim
from Revision_R2_two_rings_ME import stimulus_and_fields, TILT, sz_x, sz_y, Omega

out = current_folder+'/Toolbox/Data/work/'
sigmas = np.arange(4.0, 30.01, 0.25)


def d4g(d, s):
    r = (d/s)**2
    return (r**2-6*r+3)*np.exp(-r/2)


def centre_line_pixels(p1, p2, n=4000):
    """Ring centre line in pixel coordinates, using the mapping in make_rotating_stim
    (x, y in [-1.2, 1.2] -> index, rows indexed by y, then the image is flipped left-right)."""
    th = np.linspace(0, 2*np.pi, len(p1))
    tt = np.linspace(0, 2*np.pi, n)
    x, y = np.interp(tt, th, p1), np.interp(tt, th, p2)
    col = (sz_x-1)-(x+1.2)/2.4*(sz_x-1)
    row = (y+1.2)/2.4*(sz_y-1)
    return np.stack([row, col], 1)


rr, cc = np.mgrid[0:sz_y, 0:sz_x]
pix = np.stack([rr.ravel(), cc.ravel()], 1)


def fit_frame(img, line):
    d = cKDTree(line).query(pix)[0].reshape(sz_y, sz_x)
    y = img.ravel()
    best = None
    for s in sigmas:
        X = d4g(d, s).ravel()[:, None]
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        err = np.sum((X@coef-y)**2)
        if best is None or err < best[0]:
            best = (err, s, coef)
    err, s, (A,) = best
    fit = A*d4g(d, s)
    return fit, s, 1-err/np.sum((y-y.mean())**2)


def log_spectrum(img, pad=512):
    """Zero-padded (no mean removal, background is 0) so the image border does not wrap."""
    P = np.zeros((pad, pad)); o = (pad-sz_y)//2
    P[o:o+sz_y, o:o+sz_x] = img
    S = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(P)))+1e-3)
    return S


if __name__ == '__main__':
    orient = 'H'
    pos1, pos2, vec1, vec2, _ = stimulus_and_fields(TILT['bottom'], orient)
    video, _ = make_rotating_stim(pos1, pos2, vec1, -vec2, name=f'{orient}_D4G', sz_x=sz_x, sz_y=sz_y, scale=.01)
    fits, sig, r2 = [], [], []
    for t in range(len(video)):
        f, s, q = fit_frame(video[t], centre_line_pixels(pos1[t], pos2[t]))
        fits.append(f); sig.append(s); r2.append(q)
    fits, sig, r2 = np.array(fits), np.array(sig), np.array(r2)
    np.save(out+'D4G_fits_sigma_R2.npy', np.stack([sig, r2]))
    print(f'sigma: median {np.median(sig):.1f} px (range {sig.min():.1f}-{sig.max():.1f}); R^2: median {np.median(r2):.3f} (min {r2.min():.3f})')

    # video: original | D4G fit | residual
    frames = []
    for t in range(len(video)):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
        ax[0].imshow(video[t], cmap='gray', vmin=-.4, vmax=1.1); ax[0].set_title('Rendered ring')
        ax[1].imshow(fits[t], cmap='gray', vmin=-.4, vmax=1.1); ax[1].set_title(f'D4G fit (σ={sig[t]:.1f} px)')
        ax[2].imshow(video[t]-fits[t], cmap='RdBu_r', vmin=-.5, vmax=.5); ax[2].set_title(f'Residual (R²={r2[t]:.2f})')
        for a in ax:
            a.axis('off')
        fig.suptitle(f'Horizontal rotation, phase {np.degrees(Omega[t]):.0f}°')
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out+'R2_5_D4G_fit.mp4', frames, fps=15)

    # Fourier domain for a few frames
    show = [0, 32, 64, 96]
    fx = np.fft.fftshift(np.fft.fftfreq(512))
    fig, ax = plt.subplots(4, len(show), figsize=(4.2*len(show), 16.5))
    for c, t in enumerate(show):
        for r, (img, lab) in enumerate(((video[t], 'Rendered'), (fits[t], 'D4G fit'))):
            ax[2*r, c].imshow(img, cmap='gray', vmin=-.4, vmax=1.1)
            ax[2*r, c].set_title(f'{lab}, {np.degrees(Omega[t]):.0f}°')
            S = log_spectrum(img)
            ax[2*r+1, c].imshow(S, cmap='magma', vmin=S.max()-4, vmax=S.max(), extent=[fx[0], fx[-1], fx[0], fx[-1]])
            ax[2*r+1, c].set(xlim=[-.12, .12], ylim=[-.12, .12])
            ax[2*r+1, c].set_title(f'{lab}: log amplitude')
            circle = plt.Circle((0, 0), 1/(np.pi*sig[t]), fill=False, ls='--', color='c', lw=1.5)
            if r == 1:
                ax[2*r+1, c].add_patch(circle)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    fig.savefig(out+'R2_5_D4G_fourier.png', dpi=110)
