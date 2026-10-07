#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2. Anisotropy of the shape of the ring.
A, B: snapshots of the horizontally (Fig 1C) and vertically (Fig 1G) rotating circular rings; despite physically
identical shapes, B looks vertically elongated and narrower than A. C, D: the same illusion for two elongated diamonds,
explained by perceived angles: phi_h looks wider than phi_v although they are equal, and psi_h wider than psi_v.
Output: figures_paper/Fig2.pdf.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import numpy as np
from matplotlib.patches import Arc
import figstyle as fs

ASPECT = 2.7                                                     # diamond length / width


def diamond(ax, vertical):
    a, b = 1.0, 1/ASPECT                                         # half length, half width
    P = np.array([[-a, 0], [0, -b], [a, 0], [0, b], [-a, 0]])
    if vertical:
        P = P[:, ::-1]
    ax.plot(P[:, 0], P[:, 1], 'k', lw=1)
    acute = np.degrees(2*np.arctan(b/a)); obtuse = 180-acute
    r1, r2 = .28, .14
    if not vertical:                                             # psi_h at the left vertex, phi_v at the bottom vertex
        ax.add_patch(Arc((-a, 0), 2*r1, 2*r1, theta1=-acute/2, theta2=acute/2, lw=.8))
        ax.text(-a+r1+.06, 0, r'$\psi_h$', va='center', ha='left')
        ax.add_patch(Arc((0, -b), 2*r2, 2*r2, theta1=90-obtuse/2, theta2=90+obtuse/2, lw=.8))
        ax.text(0, -b+r2+.04, r'$\phi_v$', ha='center', va='bottom')
    else:                                                        # psi_v at the top vertex, phi_h at the left vertex
        ax.add_patch(Arc((0, a), 2*r1, 2*r1, theta1=270-acute/2, theta2=270+acute/2, lw=.8))
        ax.text(0, a-r1-.06, r'$\psi_v$', ha='center', va='top')
        ax.add_patch(Arc((-b, 0), 2*r2, 2*r2, theta1=-obtuse/2, theta2=obtuse/2, lw=.8))
        ax.text(-b+r2+.04, 0, r'$\phi_h$', va='center', ha='left')
    ax.set(xlim=[-1.1, 1.1], ylim=[-1.1, 1.1], aspect='equal'); ax.axis('off')


fig = fs.figure(50)
for i, (p, src) in enumerate((('A', 'Fig1C.mp4'), ('B', 'Fig1G.mp4'))):
    ax = fig.add_axes([.01+i*.25, .04, .23, .86]); fs.image_panel(ax, fs.video_frame(os.path.join(fs.VIDEOS, src)))
    fs.letter(ax, p, x=-.04, y=1.01)
for i, (p, v) in enumerate((('C', False), ('D', True))):
    ax = fig.add_axes([.51+i*.25, .04, .23, .86]); diamond(ax, v); fs.letter(ax, p, x=-.04, y=1.01)
fs.save(fig, 'Fig2')
