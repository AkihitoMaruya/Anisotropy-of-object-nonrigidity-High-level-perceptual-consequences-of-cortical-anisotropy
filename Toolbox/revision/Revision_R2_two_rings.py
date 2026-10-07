#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top and bottom ring for the Domini et al. (1997) comparison (Reviewer 2).

Each ring uses the single-ring model of Figure 6 (Eq. S17), centred at the origin
(the vertical offset between the rings is ignored):
    position(Omega) = R_y(Omega) . R_n(-k Omega) . p(theta)
p(theta): unit circle tilted by phi, n: its normal, R_y: rotation about the vertical
axis, R_n: spin about the ring's own normal. k=0 rigid rotation, k=1 wobbling.
The bottom ring is tilted by +phi and the top ring by -phi, as in Make_rotating_two_rings.py.
The velocities reproduce the Figure6.py velocity field exactly (checked below).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
out = current_folder + '/Toolbox/Data/work/'
os.makedirs(out, exist_ok=True)

phi = 30*np.pi/180
theta = np.linspace(0, 2*np.pi, 200, endpoint=False)


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def R_axis(n, a):
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K


def ring_motion(Omega, k, tilt, th=theta, dO=1e-6):
    """Orthographic image positions and velocities (d/dOmega) of one ring.
    Returns pos, vel with shape (len(Omega), 2, len(th)): [:,0] horizontal, [:,1] vertical."""
    P = np.stack([np.cos(tilt)*np.cos(th), np.sin(tilt)*np.cos(th), np.sin(th)])
    n = np.array([-np.sin(tilt), np.cos(tilt), 0.])
    X = lambda o: Ry(o)@R_axis(n, -k*o)@P
    pos = np.stack([X(o)[:2] for o in Omega])
    vel = np.stack([(X(o+dO)-X(o-dO))[:2]/(2*dO) for o in Omega])
    return pos, vel


def two_rings(Omega, k):
    """(bottom, top) rings, each (pos, vel)."""
    return ring_motion(Omega, k, phi), ring_motion(Omega, k, -phi)


#%% Check against the Figure6.py velocity field (bottom ring)
def fig6_vel(Om, k, th):
    v1 = np.outer(-np.sin(Om)*np.cos(k*Om)*np.cos(phi)-k*np.cos(Om)*np.sin(k*Om)*np.cos(phi)+np.cos(Om)*np.sin(k*Om)+k*np.sin(Om)*np.cos(k*Om), np.cos(th))\
        + np.outer(np.cos(Om)*np.cos(k*Om)-k*np.sin(Om)*np.sin(k*Om)-k*np.cos(k*Om)*np.cos(Om)*np.cos(phi)+np.sin(k*Om)*np.sin(Om)*np.cos(phi), np.sin(th))
    v2 = -np.sin(phi)*(np.outer(np.sin(k*Om), np.cos(th))+np.outer(np.cos(k*Om), np.sin(th)))*k
    return v1, v2


Om_chk = np.linspace(0, np.pi, 64, endpoint=False)
for k in (0, .25, .5, .75, 1):
    _, vel = ring_motion(Om_chk, k, phi)
    v1, v2 = fig6_vel(Om_chk, k, theta)
    assert np.allclose(vel[:, 0], v1, atol=1e-6) and np.allclose(vel[:, 1], v2, atol=1e-6), k

#%% Snapshots: rotation (k=0) and wobbling (k=1)
if __name__ == '__main__':
    phases = np.radians([0, 30, 60, 90, 120, 150])
    cols = {'bottom': 'b', 'top': 'r'}
    marks = [0, 50, 100, 150]                     # fixed points on each ring (like painted features)
    fig, ax = plt.subplots(2, len(phases), figsize=(4*len(phases), 8.5))
    for r, (k, name) in enumerate(((0, 'Rotation (k=0)'), (1, 'Wobbling (k=1)'))):
        rings = two_rings(phases, k)
        for c, Om in enumerate(phases):
            a = ax[r, c]
            for (pos, vel), lab in zip(rings, ('bottom', 'top')):
                a.plot(*pos[c], '-', color=cols[lab], lw=2, label=f'{lab} ring')
                a.quiver(pos[c, 0, ::10], pos[c, 1, ::10], vel[c, 0, ::10], vel[c, 1, ::10],
                         color=cols[lab], angles='xy', scale_units='xy', scale=4, width=.006)
                a.plot(pos[c, 0, marks], pos[c, 1, marks], 'o', color=cols[lab], mec='k', ms=7)
                a.plot(pos[c, 0, 0], pos[c, 1, 0], '*', color='gold', mec='k', ms=16)
            a.set(xlim=[-1.4, 1.4], ylim=[-1.4, 1.4], aspect='equal', xticks=[], yticks=[])
            a.set_title(f'{np.degrees(Om):.0f}°', fontsize=16)
        ax[r, 0].set_ylabel(name, fontsize=18)
    ax[0, 0].legend(fontsize=11, loc='lower left')
    fig.suptitle('Horizontal rotation: bottom ring (tilt +30°) and top ring (tilt −30°), centred; '
                 'arrows = velocity, dots/star = fixed points on the ring', fontsize=16)
    plt.tight_layout()
    fig.savefig(out+'R2_3_two_rings_check.png', dpi=120)

    # animation
    Om_anim = np.linspace(0, 2*np.pi, 72, endpoint=False)
    R0, R1 = two_rings(Om_anim, 0), two_rings(Om_anim, 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    arts = []
    for a, t in zip(ax, ('Rotation (k=0)', 'Wobbling (k=1)')):
        a.set(xlim=[-1.3, 1.3], ylim=[-1.3, 1.3], aspect='equal', xticks=[], yticks=[], title=t)
        arts.append([a.plot([], [], '-', color=cols[l], lw=2)[0] for l in ('bottom', 'top')]
                    + [a.plot([], [], 'o', color=cols[l], mec='k', ms=6)[0] for l in ('bottom', 'top')]
                    + [a.plot([], [], '*', color='gold', mec='k', ms=14)[0] for _ in range(2)])

    def update(i):
        for R, A in zip((R0, R1), arts):
            for j in range(2):
                p = R[j][0][i]
                A[j].set_data(p[0], p[1])
                A[2+j].set_data(p[0, marks], p[1, marks])
                A[4+j].set_data([p[0, 0]], [p[1, 0]])
        return sum(arts, [])

    FuncAnimation(fig, update, frames=len(Om_anim), blit=True).save(out+'R2_3_two_rings_check.gif', writer=PillowWriter(fps=15))
