#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary Figure S4. Orientation energy of the rim and the wobble of vertically and horizontally rotating rings.
A, C: the two rings rotating vertically (A) and horizontally (C) with a smooth contour or with rims of random dots
(8-frame lifetime) stretched along the rim into streaks (SD 16 ... 1 px) or round. B, D: model optic flow of the
anisotropic cortex (width x2.5), one ring, 45 deg phase, with the best k; B and D share one square crop (same scale).
E: best k against the rim texture, vertical (red) and horizontal (blue) rotation. Video: A-D together, two full
rotations (figures_paper/videos/FigS4.mp4). The model computes half a cycle (0-180 deg); the second half of the stimulus
is the first half turned by 180 deg in the image, and the cortex is symmetric under that turn, so the flow there is the
first half's flow turned by 180 deg. Near edge-on (about 7 deg either side) no flow is computed and no arrows are drawn. Flows: Toolbox/revision/Revision_R2_texture_spread_sweep.py (cached). Stimulus
movies: Toolbox/revision/Revision_R2_figS_two_ring_streak_movie.py [H] (cached in Toolbox/Data/work).
Interactive matching task: figures_paper/FigS4_matching_task.html (FigS4_interactive.py).
Output: figures_paper/FigS4.pdf; videos/FigS4.mp4."""
import os, sys
os.environ.setdefault('K_FIT_TRIM', '10')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import imageio.v2 as imageio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import figstyle as fs
import Revision_R2_stimulus_filters as SF
import Revision_R2_texture_spread_sweep as SW
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES
from Revision_R2_two_rings_ME import Omega
from Revision_R2_D4G_ring import d4g_ring_video
D, P = SW.D, SW.P
fs.style()
B = 2.5
OS = ('V', 'H')
ROT = {'V': 'vertically', 'H': 'horizontally'}
LEVELS = ['smooth']+SW.STREAKS
NAME = {'smooth': 'smooth ring'}; NAME.update({s: f'streaks {s:g} px' if s else 'round dots' for s in SW.STREAKS})
ORIG = {o: np.load(P.A.data+f'heeger_stimfilter_{o}.npz') for o in OS}
VID, FL, K, LP, CROP = {}, {}, {}, {}, {}
for o in OS:
    VID[o] = {'smooth': d4g_ring_video(o, sigma=1.5)}
    VID[o].update({s: SF.texture_video(o, streak=s, lifetime=SW.LIFE) for s in SW.STREAKS})
    Z = np.load(SW.cache(o))
    FL[o] = {'smooth': ORIG[o][f'original_b{B}']}; FL[o].update({s: Z[f's{s}_b{B}'] for s in SW.STREAKS})
    K[o] = {lv: D.fit3(*D.flow_phys(FL[o][lv]), o)[0] for lv in LEVELS}
    LP[o] = ring_points(o)[0]                                          # all 128 model frames (0-180 deg)
NM = len(LP['V'])                                                      # model frames per half cycle
SIDE = max(max(np.ptp(LP[o][FRAMES][..., 0]), np.ptp(LP[o][FRAMES][..., 1])) for o in OS)+60   # one square crop for B and D
for o in OS:
    CROP[o] = ((LP[o][FRAMES][..., 0].min()+LP[o][FRAMES][..., 0].max())/2, (LP[o][FRAMES][..., 1].min()+LP[o][FRAMES][..., 1].max())/2)


def model_frame(o, lv, f):
    """Stimulus image, ring points, flow (None near edge-on) and crop centre at frame f of the full cycle (0-255):
    the second half is the first half turned by 180 deg in the image (the cortex is symmetric under that turn)."""
    g, half = f % NM, f//NM
    img, Pp = VID[o][lv][g], LP[o][g]
    V = FL[o][lv][g-FRAMES[0]] if FRAMES[0] <= g <= FRAMES[-1] else None
    rc, cc = CROP[o]
    if half:
        h_, w_ = img.shape; img = img[::-1, ::-1]; Pp = np.c_[h_-1-Pp[:, 0], w_-1-Pp[:, 1]]
        V = None if V is None else -V; rc, cc = h_-1-rc, w_-1-cc
    return img, Pp, V, (rc, cc)
MOVIE = {o: imageio.get_reader(P.A.out+'R2_86_figS_two_ring_streaks'+('_H' if o == 'H' else '')+'.mp4') for o in OS}


def dir_rgb(V):
    hue = (np.arctan2(-V[..., 1], -V[..., 0])/(2*np.pi)) % 1
    return hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.full_like(hue, .9)], -1))


N = len(LEVELS); PW = .885/N                                     # panel slot width (figure fraction)


def panel(fig, j, top_mm):
    """Square panel j of a row whose top is top_mm below the figure's top."""
    W, H = fig.get_size_inches()*25.4; side = PW*.97*W
    return fig.add_axes([.01+j*PW, 1-(top_mm+side)/H, PW*.97, side/H])


def draw_rings(fig, o, f, top_mm, fsz, names=True):
    """One row at frame f of the full cycle: the flow of the anisotropic cortex on the ring for every rim texture."""
    for j, lv in enumerate(LEVELS):
        a = panel(fig, j, top_mm)
        img, Pp, V, (rc, cc) = model_frame(o, lv, f); s_ = np.percentile(np.abs(img), 99.5)
        a.imshow(img, cmap='gray', vmin=-s_, vmax=s_, alpha=.6)
        if V is not None:
            u = V/np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9); s = slice(None, None, 6)
            a.quiver(Pp[s, 1], Pp[s, 0], u[s, 0], -u[s, 1], color=dir_rgb(V[s]), angles='xy', scale_units='xy', scale=.05,
                     width=.009, headwidth=3.5, headlength=3.5)
        a.set(xlim=[cc-SIDE/2, cc+SIDE/2], ylim=[rc+SIDE/2, rc-SIDE/2]); a.axis('off')
        a.set_title((f'{NAME[lv]}\n' if names else '')+f'k = {K[o][lv]:.2f}', fontsize=fsz)
    W, H = fig.get_size_inches()*25.4; side = PW*.97*W
    ka = fig.add_axes([.89, 1-(top_mm+side*.5+9)/H, .11, 18/H]); ang = np.radians(np.arange(0, 360, 45)); v = np.stack([np.cos(ang), np.sin(ang)], 1)
    ka.quiver(np.zeros(8), np.zeros(8), v[:, 0], v[:, 1], color=dir_rgb(v), angles='xy', scale_units='xy', scale=1, width=.05, headwidth=3, headlength=3.5)
    for (x, y, ha), t in zip(((1.3, 0, 'left'), (0, 1.5, 'center'), (-1.3, 0, 'right'), (0, -1.5, 'center')), ('right', 'up', 'left', 'down')):
        ka.text(x, y, t, ha=ha, va='center', fontsize=fsz-1)
    ka.set(xlim=[-3.4, 3.4], ylim=[-2.2, 2.2], aspect='equal'); ka.axis('off')


def draw_k(ax, fsz):
    x = np.arange(len(LEVELS))
    for o in OS:
        k = [K[o][lv] for lv in LEVELS]
        ax.plot(x, k, 'o-', color=fs.COL[o], lw=1.2, ms=4.4, label=f'{"vertical" if o == "V" else "horizontal"} rotation')
    ax.set_xticks(x, [NAME[lv].replace('streaks ', '') for lv in LEVELS], fontsize=fsz-1)
    ax.set_xlabel('Rim: orientation energy concentrated along the rim  →  spread over all orientations', fontsize=fsz)
    ax.set_ylabel('Best k\n(0 rotation, 1 wobble)', fontsize=fsz); ax.set_ylim(-.05, 1.05); ax.set_yticks([0, .5, 1])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=fsz-1, frameon=False, loc='upper right')


f45 = int(np.argmin(np.abs(np.degrees(Omega)-45)))                      # model frame at 45 deg
def draw_two_rings(fig, o, f, top_mm, fsz):
    """The two-ring movie at frame f of the full cycle (7 panels, 8 px gaps, 40 px label strip), re-labelled."""
    im = MOVIE[o].get_data((f+64) % MOVIE[o].count_frames())[..., 0]    # movie frame = model frame + 64 (90 deg)
    pw = (im.shape[1]+8)//N-8
    for j, lv in enumerate(LEVELS):
        a = panel(fig, j, top_mm); a.imshow(im[40:, j*(pw+8):j*(pw+8)+pw], cmap='gray', vmin=0, vmax=255)
        a.set_title(NAME[lv].replace('smooth ring', 'smooth rings'), fontsize=fsz); a.axis('off')


def draw_rows(fig, f, fsz, hsz):
    """Rows A-D at frame f of the full cycle (the figure and every video frame)."""
    for i, (o, L1, L2) in enumerate((('V', 'A', 'B'), ('H', 'C', 'D'))):
        y = 2+70*i                                                       # row pair top (mm)
        draw_two_rings(fig, o, f, y+8, fsz); draw_rings(fig, o, f, y+43, fsz, names=False)
        for t, yy, s in ((L1, y, f'Two rings rotating {ROT[o]} (8-frame random dots on the textured rims)'),
                         (L2, y+35, f'Model optic flow, anisotropic cortex, one ring rotating {ROT[o]}')):
            H = fig.get_size_inches()[1]*25.4
            fig.text(.005, 1-yy/H, t, fontsize=hsz+2, fontweight='bold', va='top'); fig.text(.04, 1-yy/H, s, fontsize=hsz, va='top')


fig = fs.figure(190)
draw_rows(fig, f45, 7, 8)
draw_k(fig.add_axes([.12, 1-170/190, .82, 22/190]), 8)
fig.text(.005, 1-143/190, 'E', fontsize=10, fontweight='bold', va='top')
fig.text(.5, .003, 'Interactive matching task (dynamic-dot rims, adjustable orientation spread): FigS4_matching_task.html', ha='center', va='bottom', fontsize=7, color='0.35')
fs.save(fig, 'FigS4')
w = imageio.get_writer(os.path.join(fs.VOUT, 'FigS4.mp4'), fps=30, codec='libx264', quality=8, macro_block_size=1)
fig = plt.figure(figsize=(7.2, 5.6), dpi=150)                           # rows A-D (1080 x 840 px)
for f in range(4*NM):                                                   # two full rotations
    fig.clf(); draw_rows(fig, f % (2*NM), 6, 7); fig.canvas.draw(); w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
w.close(); plt.close(fig); print('saved', os.path.join(fs.VOUT, 'FigS4.mp4'))
print({o: {str(k): round(float(v), 2) for k, v in K[o].items()} for o in OS})
