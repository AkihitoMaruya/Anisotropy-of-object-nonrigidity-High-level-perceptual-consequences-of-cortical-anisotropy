#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary Figure S5. The shape of the object does not change Def / Curl.
Columns: five closed outlines on the ground plane (circle = the stimulus ring, 2:1 ellipse, octagon as Figure 1, star,
one random outline as Supplementary Figure S6), moved by Eq. S16 (ring tilt 30 deg, as the stimulus) and seen by an
orthographic camera at 0 deg elevation. Rows: A the outline; B-D Div, Curl and Def as the contour integrals S18-S20
(not divided by the area, as Figure 6B; per frame, 1.4 deg of rotation per frame); E Def / Curl as atan2(Def, Curl).
Colour: k as Figure 6. The integrals scale with the area the outline encloses, which cancels in Def / Curl: the bottom
row is the same for every shape (the flow of a rigidly moving plane is linear, so the shape only enters through its
area). Div is the same for every k.
Output: figures_paper/FigS5.pdf."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import figstyle as fs
fs.style()

TAU, W_FRAME = np.radians(30), np.radians(360/256)                  # ring tilt; rotation per frame (rad)
PH = np.radians(np.arange(1, 180, 1))
KS = np.round(np.arange(0, 1.001, .05), 2)
Y = np.array([0, 1., 0])
t = np.linspace(0, 2*np.pi, 400, endpoint=False)
o = np.linspace(0, 2*np.pi, 9)[:-1]+np.pi/8                          # octagon vertices
rng = np.random.default_rng(0); tr = np.linspace(0, 2*np.pi, 100, endpoint=False)
r1, r2 = rng.uniform(.5, 1.5, 100), rng.uniform(.5, 1.5, 100)        # the first random outline of Figure S6
SHAPES = {'Circle': (np.cos(t), np.sin(t)),                         # sizes differ so that the areas differ
          'Ellipse 2:1': (1.5*np.cos(t), .75*np.sin(t)),
          'Octagon': (.6*np.cos(o), .6*np.sin(o)),
          'Star': (1.15*(1+.4*np.cos(5*t))*np.cos(t), 1.15*(1+.4*np.cos(5*t))*np.sin(t)),
          'Random': (.8*r1*np.cos(tr), .8*r2*np.sin(tr))}


def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def invariants(X, Z, k):
    """Div, Curl and Def x |A| (per frame) and atan2(Def, Curl) over the phases, for the outline (X, 0, Z)."""
    P0 = np.stack([X, 0*X, Z]); out = np.full((4, len(PH)), np.nan)
    for j, W in enumerate(PH):
        P = Ry(W)@Rz(TAU)@Ry(-k*W)@P0
        F = np.cross(W_FRAME*(Y-k*Ry(W)@Rz(TAU)@Y), P, axis=0)        # rigid velocity, w (y - k n) x P
        x, y, u, v = P[0], P[1], F[0], F[1]
        x1, y1, u1, v1 = (np.roll(a, -1) for a in (x, y, u, v))
        dx, dy, um, vm = x1-x, y1-y, (u+u1)/2, (v+v1)/2
        A = .5*np.sum(x*y1-x1*y)
        if abs(A) < 1e-3:
            continue
        div = np.sum(um*dy-vm*dx)/A; curl = np.sum(um*dx+vm*dy)/A; dfm = np.hypot(np.sum(-um*dx+vm*dy), np.sum(um*dy+vm*dx))/abs(A)
        out[:, j] = div*abs(A), curl*abs(A), dfm*abs(A), np.degrees(np.arctan2(dfm, curl))   # S18-S20 (counterclockwise), ratio
    return out


fig = fs.figure(180)
gs = fig.add_gridspec(5, len(SHAPES), height_ratios=[.7, 1, 1, 1, 1], hspace=.5, wspace=.25, left=.1, right=.86, top=.95, bottom=.08)
cols = cm.jet(np.linspace(0, 1, len(KS))); ph = np.degrees(PH)
R = {name: np.array([invariants(*xz, k) for k in KS]) for name, xz in SHAPES.items()}
lims = [(np.nanmin([r[:, i] for r in R.values()]), np.nanmax([r[:, i] for r in R.values()])) for i in (0, 1, 2)]
for j, (name, (X, Z)) in enumerate(SHAPES.items()):
    a = fig.add_subplot(gs[0, j])
    a.fill(np.r_[X, X[:1]], np.r_[Z, Z[:1]], color='0.85', ec='k', lw=1)
    a.set(xlim=[-1.7, 1.7], ylim=[-1.7, 1.7], aspect='equal'); a.axis('off'); a.set_title(name)
    area = .5*abs(np.sum(X*np.roll(Z, -1)-np.roll(X, -1)*Z))
    a.text(0, -1.75, f'area {area/np.pi:.2f} π', ha='center', va='top', fontsize=7)
    for i, (lab, yl) in enumerate(((r'$\iint$ Div $dA$' '\n(per frame)', lims[0]), (r'$\iint$ Curl $dA$' '\n(per frame)', lims[1]),
                                   (r'$\iint$ Def $dA$' '\n(per frame)', lims[2]), (r'$\iint$ Def / Curl (deg)', (0, 180)))):
        a = fig.add_subplot(gs[i+1, j])
        for k, col, r in zip(KS, cols, R[name]):
            a.plot(ph, r[i], color=col, lw=.8)
        a.set(xlim=[0, 180], xticks=[0, 90, 180], ylim=yl if i == 3 else (yl[0]-.05*np.ptp(yl), yl[1]+.05*np.ptp(yl)))
        if i == 3:
            a.set(yticks=[0, 90, 180]); a.axhline(45, color='0.5', lw=.6, ls=':', zorder=0); a.axhline(90, color='0.5', lw=.6, ls='--', zorder=0)
        else:
            a.axhline(0, color='0.5', lw=.6, zorder=0)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        if j == 0:
            a.set_ylabel(lab); fs.letter(a, 'BCDE'[i], x=-.75, y=1.02)
        else:
            a.set_yticklabels([])
        if i < 3:
            a.set_xticklabels([])
fig.text(.48, .02, 'Motion phase (deg)', ha='center', fontsize=8)
fs.fig_letter(fig, .005, .975, 'A')
cax = fig.add_axes([.885, .1, .015, .55])
cb = fig.colorbar(cm.ScalarMappable(cmap=cm.jet), cax=cax); cb.set_label('k')
cb.set_ticks([0, .5, 1]); cb.set_ticklabels(['0 (rotation)', '0.5', '1 (wobble)'])
fs.save(fig, 'FigS5')
ref = R['Circle'][:, 3]
print('max |Def/Curl - circle| over shapes, k, phases (deg):', max(np.nanmax(np.abs(r[:, 3]-ref)) for r in R.values()))
print('Div spread over k (max over shapes, phases):', max(np.nanmax(np.ptp(r[:, 0], 0)) for r in R.values()))
