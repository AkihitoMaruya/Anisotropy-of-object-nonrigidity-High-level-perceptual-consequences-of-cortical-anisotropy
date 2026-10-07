#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render the rotating ring directly as a thin contour with a 4th-derivative-of-Gaussian (D4G)
cross-section, I(x) = D4G_sigma(d(x)), d = distance from the ring's centre line
(same centre line and pixel mapping as make_rotating_stim).

The Fourier transform of a thin ellipse is a set of concentric oval fringes (oriented
90 deg to the ring); the D4G profile multiplies it by the radial band-pass
rho^4 exp(-sigma^2 rho^2 / 2), which peaks at 1/(pi sigma) cycles/pixel. So each frame's
spectrum is one oval annulus.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.spatial import cKDTree

current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from Revision_R2_two_rings_ME import stimulus_and_fields, TILT, sz_x, sz_y, Omega
from Revision_R2_D4G_fit import d4g, centre_line_pixels, pix

out = current_folder+'/Toolbox/Data/work/'
SIGMA = 3.0                     # px; D4G band-pass peak at 1/(pi*SIGMA) ~ 0.106 cycles/px


def d4g_ring_video(orient='H', ring='bottom', sigma=SIGMA):
    pos1, pos2, _, _, _ = stimulus_and_fields(TILT[ring], orient)
    video = np.zeros((len(pos1), sz_y, sz_x))
    for t in range(len(pos1)):
        d = cKDTree(centre_line_pixels(pos1[t], pos2[t])).query(pix)[0].reshape(sz_y, sz_x)
        video[t] = d4g(d, sigma)/3               # peak 1 on the centre line
    return video


def spectrum(img):
    return np.abs(np.fft.fftshift(np.fft.fft2(img)))


if __name__ == '__main__':
    fx = np.fft.fftshift(np.fft.fftfreq(sz_x))
    lim = 0.25
    videos = {o: d4g_ring_video(o) for o in ('H', 'V')}

    # video: D4G ring and its log-amplitude spectrum, horizontal and vertical rotation
    frames = []
    for t in range(len(Omega)):
        fig, ax = plt.subplots(2, 2, figsize=(9, 9))
        for r, o in enumerate(('H', 'V')):
            ax[r, 0].imshow(videos[o][t], cmap='gray', vmin=-.5, vmax=1)
            ax[r, 0].set_title(f'{"Horizontal" if o == "H" else "Vertical"} rotation, {np.degrees(Omega[t]):.0f}°')
            S = np.log10(spectrum(videos[o][t])+1e-3)
            ax[r, 1].imshow(S, cmap='magma', vmin=S.max()-3, vmax=S.max(), extent=[fx[0], fx[-1], fx[-1], fx[0]])
            ax[r, 1].set(xlim=[-lim, lim], ylim=[-lim, lim]); ax[r, 1].set_title('log amplitude')
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([])
        plt.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out+'R2_6_D4G_ring.mp4', frames, fps=15)

    # selected frames
    show = [0, 32, 64, 96]
    fig, ax = plt.subplots(4, len(show), figsize=(4.2*len(show), 17))
    for c, t in enumerate(show):
        for r, o in enumerate(('H', 'V')):
            ax[2*r, c].imshow(videos[o][t], cmap='gray', vmin=-.5, vmax=1)
            ax[2*r, c].set_title(f'{"Horizontal" if o == "H" else "Vertical"} rot., {np.degrees(Omega[t]):.0f}°')
            S = np.log10(spectrum(videos[o][t])+1e-3)
            ax[2*r+1, c].imshow(S, cmap='magma', vmin=S.max()-3, vmax=S.max(), extent=[fx[0], fx[-1], fx[-1], fx[0]])
            ax[2*r+1, c].set(xlim=[-lim, lim], ylim=[-lim, lim])
            ax[2*r+1, c].add_patch(plt.Circle((0, 0), 1/(np.pi*SIGMA), fill=False, ls='--', color='c', lw=1))
            ax[2*r+1, c].set_title('log amplitude')
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    fig.savefig(out+'R2_6_D4G_ring_fourier.png', dpi=110)
