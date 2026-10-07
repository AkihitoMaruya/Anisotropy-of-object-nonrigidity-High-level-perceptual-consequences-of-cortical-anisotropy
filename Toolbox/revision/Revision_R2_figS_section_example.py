#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schematics of how sections are chosen for the variance of deformations (used as the top row of
Revision_R2_figS_domini_region_size.plot_methods): the two-ring contour at motion phase 90 deg, horizontal (left)
and vertical (right) rotation in one panel, with four examples (colours): 'A' global triplets (random, well-spread
triplets over both rings; each colour one triplet), 'B' local neighbours in pixel
location (dashed circle = radius, half the arc length of a 30 deg section; the 8 points used), 'C' symmetric pairs (the
same 2-D contour length, 30/360 of the image perimeter, on both rings, 4 points per ring evenly spaced in arc length
at mirror positions; circles ring 1, squares ring 2).
New analysis (geometry only)."""
import os, sys
import numpy as np
from matplotlib.patches import Circle, Polygon
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
COLS = ['#e67e22', '#27ae60', '#8e44ad', '#c0392b']
CENTRES = [0, 40, 80, 120]                                         # section centres (samples of 240): 0, 60, 120, 180 deg
W, M_PTS = 20, 8


def draw_schematic(a, method, M):
    TR, NP, FRAMES = M.TR, M.NP, M.FRAMES
    fi = int(np.argmin(np.abs(M.phase-90)))
    R_px = np.median([np.linalg.norm(np.diff(TR.two_ring_points('H')[0][FRAMES][:, :W+1], axis=1), axis=-1).sum(1)])/2
    off = 0
    for o in ('H', 'V'):
        L = TR.two_ring_points(o)[0][FRAMES][fi]
        x, y = L[:, 1], -L[:, 0]
        x = x-x.min()+off
        a.plot(np.r_[x[:NP], x[0]], np.r_[y[:NP], y[0]], color='0.75', lw=2)
        a.plot(np.r_[x[NP:], x[NP]], np.r_[y[NP:], y[NP]], color='0.75', lw=2)
        if method == 'A':                                            # a few random, well-spread triplets
            rng = np.random.default_rng(3); A = 0.5*abs(np.sum(x[:NP]*np.roll(y[:NP], -1)-np.roll(x[:NP], -1)*y[:NP])); n = 0
            while n < 4:
                t = rng.integers(0, 2*NP, 3); X, Y = x[t], y[t]
                if 0.5*abs((X[1]-X[0])*(Y[2]-Y[0])-(X[2]-X[0])*(Y[1]-Y[0])) >= TR.MIN_AREA*A:
                    a.plot(X, Y, 'o', color=COLS[n], ms=8); n += 1
        elif method == 'B':
            for col, c in zip(COLS, CENTRES):
                d = np.linalg.norm(L-L[c], axis=1); nb = np.argsort(d)[:np.sum(d <= R_px)]
                pts = nb[np.round(np.linspace(0, len(nb)-1, min(M_PTS, len(nb)))).astype(int)]
                a.add_patch(Circle((x[c], y[c]), R_px, fill=False, ec=col, lw=1.5, ls='--'))
                a.plot(x[pts], y[pts], 'o', color=col, ms=6)
        else:                                                        # equal 2-D arc length (M.symmetric_equal_arc)
            for col, c in zip(COLS, CENTRES):
                for r, mk in ((0, 'o'), (1, 's')):
                    p = np.c_[x[r*NP:(r+1)*NP], y[r*NP:(r+1)*NP]]; pc = np.vstack([p, p[:1]])
                    sarc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pc, axis=0), axis=1))]; S = sarc[-1]
                    q = (c/NP*S+np.linspace(-W/NP*S/2, W/NP*S/2, M_PTS//2)) % S
                    a.plot(np.interp(q, sarc, pc[:, 0]), np.interp(q, sarc, pc[:, 1]), mk, color=col, ms=7)
        a.text(x.mean(), y.max()+12, 'horizontal' if o == 'H' else 'vertical', ha='center', va='bottom', fontsize=16)
        off = x.max()+40
    a.set_aspect('equal'); a.axis('off')
