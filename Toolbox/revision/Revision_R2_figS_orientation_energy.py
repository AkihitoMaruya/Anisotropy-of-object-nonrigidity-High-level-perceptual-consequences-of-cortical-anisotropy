#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary figure and movie: vertically rotating ring with rims of different orientation energy, the model
optic flow of the anisotropic cortex (width anisotropy x2.5, one band, equal numbers, no smoothing) and the
estimated k. Rims: the smooth ring (D4G, sigma 1.5 px) and random-dot rims with limited lifetime (8 frames, about
270 ms, then redrawn at a random new place; Revision_R2_stimulus_filters.texture_video) whose dots are stretched
along the rim (Gaussian arc SD 16, 8, 4, 2, 1 px) or round, from orientation energy concentrated along the rim to
spread over all orientations. Flows cached by Revision_R2_texture_spread_sweep.py (V) and
Revision_R2_stimulus_filter_sweep.py (smooth ring). Bottom: best k against the rim for the anisotropic cortex,
with the smooth horizontal ring's k as reference; open markers: match quality < 0.6.
Figure: A the experimental two rings rotating vertically with each rim (still of
Revision_R2_figS_two_ring_streak_movie.py, run that first), B the model flow, C best k.
Outputs: Images/Revision_R2/R2_86_figS_orientation_energy.png and .mp4 (flow panels). New analysis (cached flows)."""
import os, sys
os.environ.setdefault('K_FIT_TRIM', '10')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import imageio.v2 as imageio
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_stimulus_filters as SF
import Revision_R2_texture_spread_sweep as SW
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES
from Revision_R2_two_rings_ME import Omega
from Revision_R2_D4G_ring import d4g_ring_video
D, P = SW.D, SW.P
plt.rcParams.update({'font.family': 'Arial', 'font.size': 20})
B, O = 2.5, 'V'
LEVELS = ['smooth']+SW.STREAKS
NAME = {'smooth': 'smooth ring'}; NAME.update({s: f'streaks {s:g} px' if s else 'round dots' for s in SW.STREAKS})
VID = {'smooth': d4g_ring_video(O, sigma=1.5)}
VID.update({s: SF.texture_video(O, streak=s, lifetime=SW.LIFE) for s in SW.STREAKS})
Z = np.load(SW.cache(O)); ORIG = {o: np.load(P.A.data+f'heeger_stimfilter_{o}.npz') for o in ('H', 'V')}
FL = {'smooth': ORIG[O][f'original_b{B}']}; FL.update({s: Z[f's{s}_b{B}'] for s in SW.STREAKS})
KQ = {lv: D.fit3(*D.flow_phys(FL[lv]), O) for lv in LEVELS}
K = {lv: KQ[lv][0] for lv in LEVELS}; M = {lv: KQ[lv][1].max() for lv in LEVELS}
R = SW.results()
LP = ring_points(O)[0][FRAMES]
r0, r1 = LP[..., 0].min()-30, LP[..., 0].max()+30; c0, c1 = LP[..., 1].min()-30, LP[..., 1].max()+30


def dir_rgb(V):
    hue = (np.arctan2(-V[..., 1], -V[..., 0])/(2*np.pi)) % 1
    return hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.full_like(hue, .9)], -1))


def draw_rings(fig, fi, top, h, fs, names=True):
    n = len(LEVELS); w = .93/n
    for j, lv in enumerate(LEVELS):
        a = fig.add_axes([.01+j*w, top-h, w*.97, h])
        img = VID[lv][FRAMES[fi]]; s_ = np.percentile(np.abs(img), 99.5)
        a.imshow(img, cmap='gray', vmin=-s_, vmax=s_, alpha=.6)
        V = FL[lv][fi]; Pp = LP[fi]; u = V/np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9); s = slice(None, None, 6)
        a.quiver(Pp[s, 1], Pp[s, 0], u[s, 0], -u[s, 1], color=dir_rgb(V[s]), angles='xy', scale_units='xy', scale=.04,
                 width=.016, headwidth=3.2, headlength=3.8)
        a.set(xlim=[c0, c1], ylim=[r1, r0]); a.axis('off')
        a.set_title((f'{NAME[lv]}\n' if names else '')+f'k = {K[lv]:.2f} (match {M[lv]:.2f})', fontsize=fs)
    ka = fig.add_axes([.935, top-h*.62, .06, h*.3]); ang = np.radians(np.arange(0, 360, 45)); v = np.stack([np.cos(ang), np.sin(ang)], 1)
    ka.quiver(np.zeros(8), np.zeros(8), v[:, 0], v[:, 1], color=dir_rgb(v), angles='xy', scale_units='xy', scale=1, width=.05, headwidth=3, headlength=3.5)
    for (x, y), t in zip(v[::2]*1.6, ('right', 'up', 'left', 'down')):
        ka.text(x, y, t, ha='center', va='center', fontsize=fs-6)
    ka.set(xlim=[-2.2, 2.2], ylim=[-2.2, 2.2], aspect='equal'); ka.axis('off')


def draw_k(ax, fs):
    x = np.arange(len(LEVELS))
    for b, col, lab in ((B, '#c0392b', 'anisotropic cortex (β 2.5 × cat)'),):
        k = [R[(O, 'orig', b)][0]]+[R[(O, s, b)][0] for s in SW.STREAKS]
        q = [R[(O, 'orig', b)][1]]+[R[(O, s, b)][1] for s in SW.STREAKS]
        ax.plot(x, k, '-', color=col, lw=3, label='vertical rotation, '+lab)
        for xi, ki, qi in zip(x, k, q):
            ax.plot(xi, ki, 'o', ms=11, mew=2.5, color=col, mfc=col if qi >= .6 else 'white')
        ax.axhline(R[('H', 'orig', b)][0], color=col, ls='--', lw=2, label=f'horizontal rotation, smooth ring, {lab}')
    ax.set_xticks(x, [NAME[lv].replace('streaks ', '') for lv in LEVELS], fontsize=fs-2)
    ax.set_xlabel('rim (orientation energy concentrated along the rim  →  spread over all orientations)', fontsize=fs)
    ax.set_ylabel('best k', fontsize=fs); ax.set_ylim(-.05, 1.05); ax.set_yticks([0, .5, 1])
    ax.text(-.04, .02, 'rotation', transform=ax.transAxes, ha='right', fontsize=fs-4, color='.35')
    ax.text(-.04, .95, 'wobble', transform=ax.transAxes, ha='right', fontsize=fs-4, color='.35')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=fs-4, frameon=False, loc='upper right')


fi45 = int(np.argmin(np.abs(np.degrees(Omega[FRAMES])-45)))
def draw_two_rings(fig, top, h, fs):
    """Panel A: the still of the two-ring movie (7 panels of 256 px, 8 px gaps, 40 px label strip), re-labelled."""
    im = imageio.imread(P.A.out+'R2_86_figS_two_ring_streaks.png'); n = len(LEVELS); w = .93/n
    pw = (im.shape[1]+8)//n-8
    for j, lv in enumerate(LEVELS):
        a = fig.add_axes([.01+j*w, top-h, w*.97, h]); a.imshow(im[40:, j*(pw+8):j*(pw+8)+pw], cmap='gray', vmin=0, vmax=255)
        a.set_title(NAME[lv].replace('smooth ring', 'smooth rings'), fontsize=fs); a.axis('off')


fig = plt.figure(figsize=(28, 21))
draw_two_rings(fig, .955, .25, 22)
draw_rings(fig, fi45, .655, .34, 20, names=False)
draw_k(fig.add_axes([.08, .05, .86, .21]), 20)
for x, y, t in ((.005, .975, 'A'), (.005, .675, 'B'), (.005, .29, 'C')):
    fig.text(x, y, t, fontsize=34, fontweight='bold', va='top')
fig.text(.04, .975, 'Two rings rotating vertically (8-frame random dots on the textured rims)', fontsize=22, va='top')
fig.text(.04, .675, 'Model optic flow, anisotropic cortex (β 2.5 × cat), one ring', fontsize=22, va='top')
fig.savefig(P.A.out+'R2_86_figS_orientation_energy.png', dpi=75); plt.close(fig)
w = imageio.get_writer(P.A.out+'R2_86_figS_orientation_energy.mp4', fps=10, codec='libx264', quality=8, macro_block_size=1)
fig = plt.figure(figsize=(21, 7.6), dpi=70)                   # even pixel size for the encoder
for fi in range(P.FIT.start, P.FIT.stop):
    fig.clf(); draw_rings(fig, fi, .88, .86, 15); fig.canvas.draw(); w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
w.close(); print({str(k): round(float(v), 2) for k, v in K.items()})
