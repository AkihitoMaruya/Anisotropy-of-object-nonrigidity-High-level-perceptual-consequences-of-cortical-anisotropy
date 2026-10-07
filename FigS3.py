#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Figure S3. Intuition for the nonrigidity anisotropy.
A-D: direction tuning of the motion-energy units and the energy they collect from the rotating ring. Coloured: each
direction unit's response to a grating at the filters' spatial frequency drifting at 0.58 px/frame in direction alpha
(its two direction-selective channels, peak 1, times the cell number; as Figure 5A, B), colour = preferred direction.
Black (right axis): the total energy each unit collects from the ring video over the whole cycle (sum over its
direction-selective channels of |video spectrum|^2 x filter^2, times the cell number) at the unit's preferred direction,
normalised to the isotropic cortex's maximum for that rotation. Columns: isotropic and anisotropic cortex (Figure 5
model); rows: horizontal (A, B) and vertical (C, D) rotation.
E, F: for one point on the ring (45 deg phase, where the anisotropy shifts the flow most), the Heeger cost over velocity
space, cost(v) = sum_i (m_i - mbar_i R_i(v)/Rbar_i(v))^2, isotropic and anisotropic cortex, with the aperture constraint
line (true normal motion), the true rigid-rotation (k = 0) and wobble (k = 1) velocities (same normal component) and the
model estimate; E horizontal, F vertical rotation.
G, H: best-fitting k (template matching, as Figure 5F/G) against the tuning-width anisotropy (cat widths scaled 0-5 x,
equal numbers) and the cell-number anisotropy (cat numbers scaled 0-5 x, equal widths); dashed: the values of the model.
Interactive 3-D view of the filters: figures_paper/FigS3_filters.html (FigS3_filters_interactive.py).
Output: figures_paper/FigS3.pdf.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
import fig5_model as M
import Revision_R2_stimulus_filters as SF
import Revision_R2_heeger_temperature_sweep as TS
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import Omega
from Revision_R2_D4G_ring import d4g_ring_video
D, P = TS.D, TS.P
fs.style()
fig = fs.figure(290)
Y0, YH = .255, .415                                                    # bottom region (E, F) in figure fractions
KX = (.935-.1)/(.922-.01)                                            # E, F: left edge aligned with A, C
B = lambda r: [.1+(r[0]-.01)*KX, Y0+r[1]*YH, r[2]*KX, r[3]*YH]
alpha = np.radians(np.arange(-180, 180, .5)); PHI_S = np.radians(30)
fg = M.f0*np.stack([np.cos(PHI_S)*np.cos(alpha), np.cos(PHI_S)*np.sin(alpha), -np.full_like(alpha, np.sin(PHI_S))], 1)
DS = M.LAYOUT_C < 0                                                   # direction-selective channels


def tuning(kind):
    b = M.bank(kind); G = np.zeros((len(b.mu), len(alpha)))
    for i, (mu, C) in enumerate(zip(b.mu, b.C)):
        G[i] = np.exp(-np.sum((fg-mu)**2, 1)/C[0, 0])+np.exp(-np.sum((-fg-mu)**2, 1)/C[0, 0])
    R = G.reshape(16, b.S, -1)[:, DS].sum(1); R = R/R.max(1, keepdims=True)
    n = M.NUM if kind == 'A' else np.full(16, M.NUM.mean())
    return n[:, None]*R, alpha[np.argmax(R, 1)]                        # responses, preferred directions


def ring_energy(kind, o):
    V = M.d4g_ring_video(o, sigma=1.5); P = np.abs(np.fft.fftshift(np.fft.fftn(V)))**2
    b = M.bank(kind); E = np.array([np.sum(P*G**2) for G in b.masks(V.shape)])
    n = M.NUM if kind == 'A' else np.full(16, M.NUM.mean())
    return n*E.reshape(16, b.S)[:, DS].sum(1)


TU = {k: tuning(k) for k in ('U', 'A')}
EN = {(k, o): ring_energy(k, o) for k in ('U', 'A') for o in ('H', 'V')}
YMAX = 1.1*max(EN[(k, o)].max()/EN[('U', o)].max() for k in ('U', 'A') for o in ('H', 'V'))
gs = fig.add_gridspec(2, 2, hspace=.6, wspace=.45, left=.1, right=.92, top=.975, bottom=.725)
for r, o in enumerate(('H', 'V')):
    norm = EN[('U', o)].max()
    for c, (kind, title) in enumerate((('U', 'Isotropic cortex'), ('A', 'Anisotropic cortex'))):
        ax = fig.add_subplot(gs[r, c]); R, pk = TU[kind]
        for k in range(16):
            ax.plot(np.degrees(alpha), R[k], color=M.dir_rgb(np.array([np.cos(pk[k]), np.sin(pk[k])])), lw=.9)
        ax.set(xticks=np.arange(-180, 181, 90), xlim=[-180, 180], ylim=[0, 1.4], xlabel='Motion direction (deg)',
               ylabel='Unit response' if c == 0 else None,
               title=f'{title}, {"horizontal" if o == "H" else "vertical"} rotation')
        a2 = ax.twinx(); i = np.argsort(pk); x = np.degrees(pk[i]); e = EN[(kind, o)][i]/norm
        a2.plot(np.r_[x[-1]-360, x, x[0]+360], np.r_[e[-1], e, e[0]], 'k-', lw=1.4)
        a2.plot(x, e, 'ko', ms=2.5)
        a2.set(ylim=[0, YMAX], ylabel='Ring energy (norm.)' if c == 1 else None)
        a2.spines['right'].set_visible(True)
        fs.letter(ax, 'ABCD'[2*r+c], x=-.2)
fi = int(np.argmin(np.abs(np.degrees(Omega[FRAMES])-45))); t = FRAMES[fi]
VM = {'H': 2.5, 'V': 0.6}                                         # axis range per row (vertical zoomed in)
for r, o in enumerate(('H', 'V')):
    L = ring_points(o)[0][FRAMES]
    vid = d4g_ring_video(o, sigma=1.5)
    Pp = L[fi]
    # the point where the anisotropy changes the flow most in the perceived direction (cached Figure-5-like flows,
    # width anisotropy 2.5, equal numbers, Revision_R2_width_vs_number_sweep.py): horizontal rotation -> towards rigid,
    # vertical rotation -> towards wobble
    Z = np.load(P.A.data+f'heeger_width_vs_number_{o}.npz')
    R0 = -np.stack(D.unit(*D.tmpl_vel(0, o)), -1)[fi]
    ang = lambda v: np.degrees(np.arccos(np.clip((v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)*R0).sum(1), -1, 1)))
    shift = ang(Z['w2.5'][fi])-ang(Z['w0.0'][fi])
    ip = int(np.argmin(shift)) if o == 'H' else int(np.argmax(shift))
    pts = (np.array([t]), np.array([int(round(Pp[ip, 0]))]), np.array([int(round(Pp[ip, 1]))]))
    tan = Pp[(ip+1) % NP]-Pp[ip-1]; tan = np.array([tan[1], -tan[0]]); tan /= np.linalg.norm(tan)  # screen (x, y up)
    nrm = np.array([-tan[1], tan[0]])
    A = P.A                                                           # true template velocities in screen px/frame
    vel = lambda k: -np.array([x[fi, ip] for x in D.tmpl_vel(k, o)])*A.dO*A.PX
    T0, T1 = vel(0), vel(1)                                           # rigid and wobble: same normal component
    a0 = fig.add_axes(B([.01, .62-.5*r, .2, .3]))
    ring = np.vstack([Pp, Pp[:1]])                                     # schematic: the ring outline at this phase
    a0.plot(ring[:, 1], ring[:, 0], 'k-', lw=1.6); a0.plot(Pp[ip, 1], Pp[ip, 0], 'o', mfc='none', mec='r', ms=10.4, mew=1.6)
    a0.set(xlim=[0, vid.shape[2]], ylim=[vid.shape[1], 0], aspect='equal')
    a0.set_title(f'{"Horizontal" if o == "H" else "Vertical"} rotation, 45°'); a0.axis('off')
    fs.letter(a0, 'EF'[r], x=-.42, y=1.12)
    for c, (kind, beta) in enumerate((('isotropic', 0), ('anisotropic', 2.5))):
        b = SF.bank(beta); M = b.energies(vid, pts)
        g, VX, VY = b.velocity_grid(); R = b.predicted(VX.ravel(), VY.ravel())
        cost = b.cost(M, R)[0].reshape(VX.shape); i = np.argmin(cost); est = np.array([VX.ravel()[i], VY.ravel()[i]])
        a = fig.add_axes(B([.29+.33*c, .62-.5*r, .26, .3]))
        cn = (cost-cost.min())/(np.percentile(cost, 5)-cost.min())        # 0 at the minimum, 1 at the 5th percentile
        im = a.imshow(np.clip(cn, 0, 1), extent=[g[0], g[-1], g[0], g[-1]], origin='lower', cmap='viridis_r', vmin=0, vmax=1,
                      interpolation='bilinear')
        a.contour(VX, VY, cn, levels=[.1, .3, .6], colors='w', linewidths=0.48, alpha=.7)
        s_n = T0@nrm                                                  # the edge's true normal speed (same for k = 0 and 1)
        line = np.outer(np.linspace(-6, 6, 2), tan)+s_n*nrm
        a.plot(line[:, 0], line[:, 1], 'w--', lw=1.2, label='aperture constraint line (true normal motion)')
        sp = np.linalg.norm(est)
        VMAX = VM[o]
        for vec, col, nm in ((T0, 'k', 'rigid rotation (k = 0), true velocity'), (T1, '0.6', 'wobble (k = 1), true velocity')):
            L_ = np.linalg.norm(vec); tip = vec if L_ <= .92*VMAX else vec/L_*.92*VMAX     # long arrows cut at the frame
            a.annotate('', xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', lw=2, color=col, mutation_scale=13.6))
            a.plot([], [], color=col, lw=1.6, label=nm)
        a.annotate('', xy=est, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', lw=2.4, color='r', mutation_scale=14.4))
        a.plot([], [], color='r', lw=2, label='model estimate')
        a.set(xlim=[-VMAX, VMAX], ylim=[-VMAX, VMAX], aspect='equal', xlabel=r'$v_x$ (px/frame)', ylabel=r'$v_y$ (px/frame)',
              title=f'{kind.capitalize()} cortex')
        if r == 0 and c == 0:
            hl = a.get_legend_handles_labels()
        ang = np.degrees(np.arccos(np.clip(est@T0/max(sp*np.linalg.norm(T0), 1e-9), -1, 1)))
        a.text(.98, .02, f'estimate {ang:.0f}° from rigid', transform=a.transAxes, ha='right', va='bottom', fontsize=6.8, color='k',
               bbox=dict(fc='w', ec='none', alpha=.8))
cb = fig.add_axes(B([.91, .3, .012, .5])); plt.colorbar(im, cax=cb, label='cost (0 = minimum)')
fig.text(.1+(.55-.01)*KX, Y0+YH+.02, r'Cost $\sum_i (m_i - \bar m_i R_i(v)/\bar R_i(v))^2$ over velocity space', ha='center', va='top')
from matplotlib.lines import Line2D
hl = ([Line2D([], [], color='0.25', ls='--', lw=1.2)]+hl[0][1:], hl[1])            # dashed line visible on white
fig.legend(*hl, loc='lower center', ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(.1+(.55-.01)*KX, Y0-.03))
# G, H: strength of the anisotropy and the best k (template matching; cached flows of Revision_R2_width_vs_number_sweep.py)
LEVELS = np.round(np.arange(0, 5.01, .5), 1)
SW = {o: np.load(P.A.data+f'heeger_width_vs_number_{o}.npz') for o in ('H', 'V')}
KS = {(o, kd, lv): D.fit3(*D.flow_phys(SW[o][f'{kd}{lv}']), o)[0] for o in ('H', 'V') for kd in ('w', 'n') for lv in LEVELS}
for j, (kd, lab, xl, used) in enumerate((('w', 'G', 'Tuning-width anisotropy (× cat), equal numbers', 2.5),
                                         ('n', 'H', 'Cell-number anisotropy (× cat), equal widths', 1.0))):
    ax = fig.add_axes([.1+j*.47, .065, .36, .125])
    for o in ('V', 'H'):
        ax.plot(LEVELS, [KS[(o, kd, l)] for l in LEVELS], 'o-', color=fs.COL[o], label='vertical rotation' if o == 'V' else 'horizontal rotation')
    ax.axvline(used, color='0.4', ls='--', lw=.8); ax.text(used+.08, 1.04, 'model', color='0.35', fontsize=7)
    ax.set(xticks=range(6), ylim=[0, 1.12], xlabel=xl, ylabel='Best k' if j == 0 else None)
    fs.letter(ax, lab, x=-.22)
    if j == 1:
        ax.legend(loc='lower right')
fig.text(.5, .004, 'Interactive 3-D view of the filters with adjustable anisotropy: FigS3_filters.html', ha='center', va='bottom', fontsize=7, color='0.35')
fs.save(fig, 'FigS3')
