#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 6. Gradients of motion-energy flow from the rotating ring.
A: differential-invariant fields (divergence, curl, deformation 1 and 2).
B: gradients of rotation to wobbling (k = 0 blue to 1 red): Div, Curl, Def from the closed-form contour integrals
   (Toolbox/ring_invariants.py; Supplementary Eqs. S18-S20), not divided by the area (the area cancels in Def / Curl).
C: Def / Curl (atan2(def, curl)) of the gradients. The templates fitted in D-H are the same closed forms (with the
   90 deg turn for vertical rotation and S J S^-1 for the image stretch).
D, E: Def / Curl of the model flows (Figure 5 model) and the best-fitting gradient, isotropic and anisotropic cortex,
   vertical (red) and horizontal (blue) rotation; shaded: deviation of the model flow from the fit.
F: vertical - horizontal best fit (anisotropic: solid; isotropic: dotted), largest difference marked.
G, H: best k against horizontal image stretch with least-squares lines, isotropic and anisotropic cortex.
Model flows: Toolbox/revision/Revision_R2_fig6CD_revised.py (Figure 5 flows), Revision_R2_fig6_stretch.py (stretched
flows), cached in Toolbox/Data/Rings. Output: figures_paper/Fig6.pdf.
"""
import os, sys
os.environ.setdefault('K_FIT_TRIM', '10')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import figstyle as fs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Revision_R2_fig6B_revised as B
import Revision_R2_fig6CD_revised as CD
import Revision_R2_fig6_stretch as ST
import ring_invariants as RI
from Revision_R2_two_rings_ME import Omega
fs.style()
P = CD.P
COL = CD.COL
STRETCHES = np.round(np.linspace(1, 1.5, 11), 2)
ph = B.phase
W = np.radians(ph)                                                   # model frames' phases (rad)
NORMALISE = False                                                    # B: contour integrals (area-weighted); True: divided by the enclosed area
TPL = {o: np.array([RI.def_curl_deg(RI.orient(RI.jacobian(k, W), o)) for k in CD.KG]) for o in ('H', 'V')}


def best_k(r, T):
    """Least-squares fit of the Def / Curl angle over all phases."""
    return CD.KG[np.nansum((T[:, CD.FIT]-r[None, CD.FIT])**2, 1).argmin()]
XT = dict(xticks=[0, 45, 90, 135, 180], xlim=[0, 180])


def clean(a):
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)


def letter(a, s, title=''):
    fs.letter(a, s, x=-.36, title=title or None)


fig = fs.figure(225)
gs = fig.add_gridspec(4, 3, height_ratios=[.62, 1, 1, 1], hspace=.6, wspace=.6, left=.09, right=.88, top=.97, bottom=.05)

# A: invariant fields
gA = gs[0, :].subgridspec(1, 4, wspace=.35)
g = np.array([-1, 0, 1]); X, Y = np.meshgrid(g, g)
fields = (('Divergence', X, Y), ('Curl', Y, -X), ('Deformation 1', Y, X), ('Deformation 2', X, -Y))
for i, (name, U, V) in enumerate(fields):
    a = fig.add_subplot(gA[i]); m = ~((X == 0) & (Y == 0))
    a.quiver(X[m]-U[m]*.3, Y[m]-V[m]*.3, U[m], V[m], angles='xy', scale_units='xy', scale=1/.6, width=.025, headwidth=4, color='k')
    a.plot(0, 0, 'ko', ms=2)
    a.set(xlim=[-1.7, 1.7], ylim=[-1.7, 1.7], aspect='equal', xticks=[], yticks=[])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_linewidth(.8)
    a.set_xlabel(name, fontsize=8)
    if i == 0:
        a.text(-.35, 1.08, 'A', transform=a.transAxes, fontsize=10, fontweight='bold', va='bottom')

# B: Div, Curl, Def;  C: Def / Curl templates
cols = cm.jet(np.linspace(0, 1, len(B.KS)))
aB = [fig.add_subplot(gs[1, c]) for c in range(3)]; aC = fig.add_subplot(gs[2, 0])
for k, col in zip(B.KS, cols):
    d_, cu_, _, _, df_ = RI.invariants(RI.jacobian(k, W, omega=B.A.dO))      # per frame
    T = [d_, cu_, df_] if NORMALISE else [v*np.abs(RI.area(W)) for v in (d_, cu_, df_)]
    for a, v in zip(aB, T):
        a.plot(ph, v, color=col, lw=1)
    aC.plot(ph, np.degrees(np.arctan2(T[2], T[1])), color=col, lw=1)
for a, lab in zip(aB, [f'{n} (per frame)' if NORMALISE else rf'$\iint$ {n} $dA$ (per frame)' for n in ('Div', 'Curl', 'Def')]):
    a.axhline(0, color='0.5', lw=.6, zorder=0); a.set(ylabel=lab, xlabel='Motion phase (deg)', **XT); clean(a)
letter(aB[0], 'B', 'Gradients of rotation to wobbling')
aC.axhline(45, color='0.5', lw=.6, ls=':', zorder=0); aC.axhline(90, color='0.5', lw=.6, ls='--', zorder=0)
aC.set(ylim=[0, 180], yticks=[0, 45, 90, 135, 180], ylabel=r'$\iint$ Def / Curl (deg)', xlabel='Motion phase (deg)', **XT); clean(aC)
aC.text(3, 92, 'curl = 0', fontsize=6, color='0.4', va='bottom', bbox=dict(fc='white', ec='none', pad=1), zorder=5); aC.text(3, 41, 'def = curl', fontsize=6, color='0.4', va='top')
letter(aC, 'C', 'Def / Curl of the gradients')
pb = aB[2].get_position(); cax = fig.add_axes([pb.x1+.015, pb.y0, .01, pb.height])   # row B only
cb = fig.colorbar(cm.ScalarMappable(cmap=cm.jet), cax=cax); cb.set_label('k')
cb.set_ticks([0, .5, 1]); cb.set_ticklabels(['0 (rotation)', '0.5', '1 (wobble)'])

# D, E: model flows and best fits;  F: vertical - horizontal
F = {o: np.load(P.A.data+f'heeger_fig5_hs_{o}.npz') for o in ('H', 'V')}
FIT, EMP = {}, {}
for a, kind, lab, title in ((fig.add_subplot(gs[2, 1]), 'U', 'D', 'Isotropic cortex'), (fig.add_subplot(gs[2, 2]), 'A', 'E', 'Anisotropic cortex')):
    for o in ('V', 'H'):
        r = CD.flow_ratio(F[o][f'{kind}_a1.0'], o); k = best_k(r, TPL[o])
        FIT[(kind, o)] = TPL[o][list(CD.KG).index(k)]; EMP[(kind, o)] = r
        a.fill_between(ph, r, FIT[(kind, o)], color=COL[o], alpha=.25 if kind == 'A' else .18, lw=0)
        a.plot(ph, FIT[(kind, o)], color=COL[o], lw=1.4, alpha=1 if kind == 'A' else .55,
               label=f'{"vertical" if o == "V" else "horizontal"}, best k = {k:.2f}')
    a.axhline(45, color='0.5', lw=.6, ls=':', zorder=0); a.axhline(90, color='0.5', lw=.6, ls='--', zorder=0)
    a.set(ylim=[0, 180], yticks=[0, 45, 90, 135, 180], ylabel=r'$\iint$ Def / Curl (deg)', xlabel='Motion phase (deg)', **XT); clean(a)
    a.legend(fontsize=6, frameon=False, loc='upper left'); letter(a, lab, title)
aF = fig.add_subplot(gs[3, 0])
for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
    d = FIT[(kind, 'V')]-FIT[(kind, 'H')]
    aF.fill_between(ph, EMP[(kind, 'V')]-EMP[(kind, 'H')], d, color='k', alpha=.15, lw=0)
    aF.plot(ph, d, ls, color='k', lw=1.4, label=cl)
d = FIT[('A', 'V')]-FIT[('A', 'H')]; i = int(np.argmax(d))
aF.axvline(ph[i], color='0.4', lw=.8, ls='--', zorder=0)
aF.text(ph[i]+5, 70, f'largest difference\nat {ph[i]:.0f}° ({d[i]:.0f}°)', fontsize=6, ha='left', va='bottom')
aF.set(ylim=[-5, 92], ylabel='Vertical − horizontal\nbest fit (deg)', xlabel='Motion phase (deg)', **XT); clean(aF)
aF.legend(fontsize=6, frameon=False, loc='lower left', bbox_to_anchor=(0, .12)); letter(aF, 'F', 'Nonrigidity anisotropy')

# G, H: stretch
Wf = Omega[ST.FRAMES]; R = {}
for o in ('H', 'V'):
    Z = np.load(P.A.data+f'heeger_fig6_stretch_{o}.npz')
    for st in STRETCHES:
        T = np.array([RI.def_curl_deg(RI.stretch(RI.orient(RI.jacobian(k, Wf), o), st)) for k in ST.KG])
        for kind in ('U', 'A'):
            R[(o, kind, st)] = ST.best_k(ST.green_ratio(ST.stretched_points(o, st), Z[f'{kind}_s{st}']), T)
x = (STRETCHES-1)*100
for c, kind, lab, title in ((1, 'U', 'G', 'Isotropic cortex'), (2, 'A', 'H', 'Anisotropic cortex')):
    a = fig.add_subplot(gs[3, c])
    for o in ('V', 'H'):
        kk = np.array([R[(o, kind, s)] for s in STRETCHES]); sl, ic = np.polyfit(x, kk, 1)   # least-squares line
        a.plot(x, kk, 'o', color=COL[o], ms=3)
        a.plot(x, ic+sl*x, '-', color=COL[o], lw=1.4,
               label=f'{"vertical" if o == "V" else "horizontal"} rotation')
    a.set(xlabel='Horizontal image stretch (%)', xticks=[0, 10, 20, 30, 40, 50], ylim=[.1, 1.0], yticks=[.2, .4, .6, .8, 1], ylabel='Best k (0 rotation, 1 wobble)')
    clean(a); letter(a, lab, title)
    a.legend(frameon=False, loc='lower left', fontsize=6)
fs.save(fig, 'Fig6')
