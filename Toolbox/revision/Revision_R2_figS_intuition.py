#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary, intuition: how the tuning-width anisotropy shifts the local velocity estimate. For one point on the
ring (45 deg phase, one point on the long upper edge and one on the side), the Heeger cost over velocity space
(isotropic vs anisotropic cortex, width anisotropy beta = 2.5, one band f0 = 0.12) with the aperture constraint line.
Rigid rotation and wobble move the contour identically: their true velocities share the normal component and differ
only along the contour, so both end on the aperture line, and the estimate's position along that line decides k.
The cost is Heeger's: cost(v) = sum_i (m_i - mbar_i R_i(v) / Rbar_i(v))^2, m_i the measured motion energy of filter i,
R_i(v) its predicted energy for velocity v, mbar_i and Rbar_i their sums over the 10 filters of filter i's spatial axis.
(the velocities consistent with the edge's normal motion), the rigid-rotation (k = 0) and wobble (k = 1) directions and
the model's estimate. Horizontal and vertical rotation. New analysis.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_stimulus_filters as SF
import Revision_R2_heeger_temperature_sweep as TS
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import Omega
from Revision_R2_D4G_ring import d4g_ring_video
D, P = TS.D, TS.P
plt.rcParams.update({'font.family': 'Arial', 'font.size': 20})
fi = int(np.argmin(np.abs(np.degrees(Omega[FRAMES])-45))); t = FRAMES[fi]
VM = {'H': 2.5, 'V': 0.6}                                         # axis range per row (vertical zoomed in)
fig = plt.figure(figsize=(24, 14))
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
    a0 = fig.add_axes([.01, .55-.5*r, .22, .39])
    a0.imshow(vid[t], cmap='gray', vmin=-1.5, vmax=1.5, alpha=.6); a0.plot(Pp[ip, 1], Pp[ip, 0], 'o', mfc='none', mec='r', ms=26, mew=4)
    a0.set_title(f'{"Horizontal" if o == "H" else "Vertical"} rotation, 45°'); a0.axis('off')
    for c, (kind, beta) in enumerate((('isotropic', 0), ('anisotropic', 2.5))):
        b = SF.bank(beta); M = b.energies(vid, pts)
        g, VX, VY = b.velocity_grid(); R = b.predicted(VX.ravel(), VY.ravel())
        cost = b.cost(M, R)[0].reshape(VX.shape); i = np.argmin(cost); est = np.array([VX.ravel()[i], VY.ravel()[i]])
        a = fig.add_axes([.27+.35*c, .55-.5*r, .31, .39])
        cn = (cost-cost.min())/(np.percentile(cost, 5)-cost.min())        # 0 at the minimum, 1 at the 5th percentile
        im = a.imshow(np.clip(cn, 0, 1), extent=[g[0], g[-1], g[0], g[-1]], origin='lower', cmap='viridis_r', vmin=0, vmax=1,
                      interpolation='bilinear')
        a.contour(VX, VY, cn, levels=[.1, .3, .6], colors='w', linewidths=1.2, alpha=.7)
        s_n = T0@nrm                                                  # the edge's true normal speed (same for k = 0 and 1)
        line = np.outer(np.linspace(-6, 6, 2), tan)+s_n*nrm
        a.plot(line[:, 0], line[:, 1], 'w--', lw=3, label='aperture constraint line (true normal motion)')
        sp = np.linalg.norm(est)
        VMAX = VM[o]
        for vec, col, nm in ((T0, 'k', 'rigid rotation (k = 0), true velocity'), (T1, '0.6', 'wobble (k = 1), true velocity')):
            L_ = np.linalg.norm(vec); tip = vec if L_ <= .92*VMAX else vec/L_*.92*VMAX     # long arrows cut at the frame
            a.annotate('', xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', lw=5, color=col, mutation_scale=34))
            if L_ > .92*VMAX:
                a.text(tip[0]-.06*VMAX, tip[1]-.06*VMAX, f'{L_:.1f} px/frame\n(cut)', color=col, fontsize=14, ha='right', va='top',
                       bbox=dict(fc='w', ec='none', alpha=.7))
            a.plot([], [], color=col, lw=4, label=nm)
        a.annotate('', xy=est, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', lw=6, color='r', mutation_scale=36))
        a.plot([], [], color='r', lw=5, label='model estimate')
        a.set(xlim=[-VMAX, VMAX], ylim=[-VMAX, VMAX], aspect='equal', xlabel=r'$v_x$ (px/frame)', ylabel=r'$v_y$ (px/frame)',
              title=f'{kind} cortex: ' + r'$\sum_i (m_i - \bar m_i R_i(v)/\bar R_i(v))^2$')
        if r == 0 and c == 0:
            hl = a.get_legend_handles_labels()
        ang = np.degrees(np.arccos(np.clip(est@T0/max(sp*np.linalg.norm(T0), 1e-9), -1, 1)))
        a.text(.98, .02, f'estimate {ang:.0f}° from rigid', transform=a.transAxes, ha='right', va='bottom', fontsize=17, color='k',
               bbox=dict(fc='w', ec='none', alpha=.8))
cb = fig.add_axes([.94, .3, .012, .4]); plt.colorbar(im, cax=cb, label='cost (0 = minimum)')
from matplotlib.lines import Line2D
hl = ([Line2D([], [], color='0.25', ls='--', lw=3)]+hl[0][1:], hl[1])            # dashed line visible on white
fig.legend(*hl, loc='upper center', ncol=2, fontsize=18, frameon=False, bbox_to_anchor=(.5, -.02))
fig.savefig(P.A.out+'R2_86_figS_intuition.png', dpi=70, bbox_inches='tight'); print('ok')
