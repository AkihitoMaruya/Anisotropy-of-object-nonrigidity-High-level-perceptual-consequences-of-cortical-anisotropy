#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary Figure S6. Def / Curl of rotation to wobbling does not change when the image is rotated.
The ring P(theta) = (cos theta, 0, sin theta) moves by P_k = R_y(wt) R_Z(tau) R_y(-k wt) P(theta) (Supplementary Eqs. S4-S5,
S16) and is projected orthographically; the projected positions and velocities are then rotated in the image plane by
alpha (alpha = 90 deg: vertical rotation). Div, Curl and Def from the contour integrals S18-S20 around the projected ring,
counterclockwise; atan2(Def, Curl) per motion phase. Columns: image rotation alpha; rows: ring tilt tau; colour: k (as
Figure 6). Gaps: the ring edge-on.
Output: figures_paper/FigS6.pdf."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import figstyle as fs
fs.style()

TH = np.linspace(0, 2*np.pi, 400, endpoint=False)
RING = np.stack([np.cos(TH), 0*TH, np.sin(TH)])                      # the ring on the ground plane
PH = np.radians(np.arange(1, 180, 1))                                # motion phase wt (0-180 deg)
KS = np.round(np.arange(0, 1.001, .05), 2)                           # as Figure 6B
ALPHAS, TAUS = (0, 45, 90, 135), (15, 30, 45, 60)
Y = np.array([0, 1., 0])


def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def Q(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s], [s, c]])


def def_curl(x, y, u, v):
    """atan2(Def, Curl) (deg) from S18-S20 around the closed polygon; divided by the signed area (orientation-free)."""
    x1, y1, u1, v1 = (np.roll(a, -1) for a in (x, y, u, v))
    dx, dy, um, vm = x1-x, y1-y, (u+u1)/2, (v+v1)/2
    A = .5*np.sum(x*y1-x1*y)
    curl = np.sum(um*dx+vm*dy)/A
    dfm = np.hypot(np.sum(-um*dx+vm*dy), np.sum(um*dy+vm*dx))/abs(A)
    return np.degrees(np.arctan2(dfm, curl))


def angles(alpha, tau, k):
    """atan2(Def, Curl) over the phases for the ring, image rotated by alpha; NaN where the ring is (nearly) edge-on."""
    R0, Qa = Rz(np.radians(tau)), Q(np.radians(alpha)); out = np.full(len(PH), np.nan)
    for j, W in enumerate(PH):
        if abs((Ry(W)@R0@Y)[2]) < .08:
            continue
        P = Ry(W)@R0@Ry(-k*W)@RING
        F = np.cross((Y-k*Ry(W)@R0@Y)[:, None], P, axis=0)           # rigid velocity, w (y - k n) x P, w = 1
        xy, uv = Qa@P[:2], Qa@F[:2]                                  # rotate the image
        out[j] = def_curl(xy[0], xy[1], uv[0], uv[1])
    return out


fig = fs.figure(175)
gs = fig.add_gridspec(len(TAUS), len(ALPHAS), hspace=.45, wspace=.3, left=.1, right=.86, top=.94, bottom=.07)
cols = cm.jet(np.linspace(0, 1, len(KS))); ph = np.degrees(PH)
R = {(a_, t_): np.array([angles(a_, t_, k) for k in KS]) for a_ in ALPHAS for t_ in TAUS}
for i, tau in enumerate(TAUS):
    for j, alpha in enumerate(ALPHAS):
        a = fig.add_subplot(gs[i, j])
        for r, col in zip(R[(alpha, tau)], cols):
            a.plot(ph, r, color=col, lw=.8)
        a.axhline(45, color='0.5', lw=.6, ls=':', zorder=0); a.axhline(90, color='0.5', lw=.6, ls='--', zorder=0)
        a.set(xlim=[0, 180], ylim=[0, 180], xticks=[0, 90, 180], yticks=[0, 90, 180])
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        if i == 0:
            a.set_title(f'Image rotation {alpha}°' + ('\n(horizontal rotation)' if alpha == 0 else '\n(vertical rotation)' if alpha == 90 else '\n'))
        if j == 0:
            a.set_ylabel(f'Ring tilt τ = {tau}°\n' + r'$\iint$ Def / Curl (deg)')
        else:
            a.set_yticklabels([])
        if i == len(TAUS)-1:
            a.set_xlabel('Motion phase (deg)')
        else:
            a.set_xticklabels([])
cax = fig.add_axes([.885, .3, .015, .45])
cb = fig.colorbar(cm.ScalarMappable(cmap=cm.jet), cax=cax); cb.set_label('k')
cb.set_ticks([0, .5, 1]); cb.set_ticklabels(['0 (rotation)', '0.5', '1 (wobble)'])
fs.save(fig, 'FigS6')
print('max |difference from image rotation 0| (deg):', max(np.nanmax(np.abs(R[(a_, t_)]-R[(0, t_)])) for a_ in ALPHAS for t_ in TAUS))
