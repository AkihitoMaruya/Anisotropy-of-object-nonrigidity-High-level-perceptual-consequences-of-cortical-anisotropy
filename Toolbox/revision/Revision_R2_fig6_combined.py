#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revised Figure 6, drawn as one figure (three equal columns from row B down):
A  the differential-invariant fields (divergence, curl, deformation 1 and 2), redrawn on white.
B  gradients of rotation to wobbling: Div, Curl, Def (contour integral / enclosed area; Revision_R2_fig6B_revised).
C  Def / Curl (atan2(def, curl)) of the rotation-to-wobbling templates.
D, E  Def / Curl of the model flows (Figure 5 model) and the best-fitting template, isotropic and anisotropic cortex,
   vertical (red) and horizontal (blue) rotation; shaded: deviation of the model flow from the fit
   (Revision_R2_fig6CD_revised).
F  vertical - horizontal best fit (anisotropic: solid; isotropic: dotted), largest difference marked.
G, H  best k against horizontal image stretch (dots) with least-squares lines, isotropic and anisotropic cortex
   (Revision_R2_fig6_stretch.py).
Output: Images/Revision_R2/R2_86_fig6_combined.png. New analysis (cached flows)."""
import os, sys
os.environ.setdefault('K_FIT_TRIM', '10')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_fig6B_revised as B
import Revision_R2_fig6CD_revised as CD
P = CD.P
plt.rcParams.update({'font.family': 'Arial', 'font.size': 26, 'axes.linewidth': 1.8,
                     'xtick.major.width': 1.8, 'ytick.major.width': 1.8, 'xtick.major.size': 7, 'ytick.major.size': 7})
COL = CD.COL
STRETCHES = np.round(np.linspace(1, 1.5, 11), 2)
ph = B.phase
XT = dict(xticks=[0, 45, 90, 135, 180], xlim=[0, 180])


def clean(a):
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)


def letter(a, s, title=''):
    a.set_title(f'{s}   {title}' if title else s, loc='left', fontweight='bold', fontsize=30)


fig = plt.figure(figsize=(30, 33))
gs = fig.add_gridspec(4, 3, height_ratios=[.62, 1, 1, 1], hspace=.45, wspace=.32, left=.06, right=.88, top=.97, bottom=.04)

# A: invariant fields
gA = gs[0, :].subgridspec(1, 4, wspace=.35)
g = np.array([-1, 0, 1]); X, Y = np.meshgrid(g, g)
fields = (('Divergence', X, Y), ('Curl', Y, -X), ('Deformation 1', Y, X), ('Deformation 2', X, -Y))
for i, (name, U, V) in enumerate(fields):
    a = fig.add_subplot(gA[i]); m = ~((X == 0) & (Y == 0))
    a.quiver(X[m]-U[m]*.3, Y[m]-V[m]*.3, U[m], V[m], angles='xy', scale_units='xy', scale=1/.6, width=.02, headwidth=4, color='k')
    a.plot(0, 0, 'ko', ms=6)
    a.set(xlim=[-1.7, 1.7], ylim=[-1.7, 1.7], aspect='equal', xticks=[], yticks=[])
    for sp in a.spines.values():
        sp.set_linewidth(1.5)
    a.set_xlabel(name, fontsize=28)
    if i == 0:
        a.text(-.35, 1.08, 'A', transform=a.transAxes, fontsize=30, fontweight='bold', va='bottom')

# B: Div, Curl, Def;  C: Def / Curl templates
cols = cm.jet(np.linspace(0, 1, len(B.KS)))
aB = [fig.add_subplot(gs[1, c]) for c in range(3)]; aC = fig.add_subplot(gs[2, 0])
for k, col in zip(B.KS, cols):
    T = B.per_area_templates(k)
    for a, v in zip(aB, T):
        a.plot(ph, v, color=col, lw=3)
    aC.plot(ph, np.degrees(np.arctan2(T[2], T[1])), color=col, lw=3)
for a, lab in zip(aB, ('Div (per frame)', 'Curl (per frame)', 'Def (per frame)')):
    a.axhline(0, color='0.5', lw=1.2, zorder=0); a.set(ylabel=lab, xlabel='Motion phase (deg)', **XT); clean(a)
letter(aB[0], 'B', 'Gradients of rotation to wobbling')
aC.axhline(45, color='0.5', lw=1.2, ls=':', zorder=0); aC.axhline(90, color='0.5', lw=1.2, ls='--', zorder=0)
aC.set(ylim=[0, 180], yticks=[0, 45, 90, 135, 180], ylabel='Def / Curl (deg)', xlabel='Motion phase (deg)', **XT); clean(aC)
aC.text(3, 92, 'curl = 0', fontsize=20, color='0.4', va='bottom', bbox=dict(fc='white', ec='none', pad=1), zorder=5); aC.text(3, 41, 'def = curl', fontsize=20, color='0.4', va='top')
letter(aC, 'C', 'Def / Curl of the gradients')
pb = aB[2].get_position(); cax = fig.add_axes([pb.x1+.015, pb.y0, .012, pb.height])   # row B only
cb = fig.colorbar(cm.ScalarMappable(cmap=cm.jet), cax=cax); cb.set_label('k')
cb.set_ticks([0, .5, 1]); cb.set_ticklabels(['0 (rotation)', '0.5', '1 (wobble)'])

# D, E: model flows and best fits;  F: vertical - horizontal
F = {o: np.load(P.A.data+f'heeger_fig5_hs_{o}.npz') for o in ('H', 'V')}
FIT, EMP = {}, {}
for a, kind, lab, title in ((fig.add_subplot(gs[2, 1]), 'U', 'D', 'Isotropic cortex'), (fig.add_subplot(gs[2, 2]), 'A', 'E', 'Anisotropic cortex')):
    for o in ('V', 'H'):
        r = CD.flow_ratio(F[o][f'{kind}_a1.0'], o); k = CD.best_k(r, o)
        FIT[(kind, o)] = CD.TPL[o][list(CD.KG).index(k)]; EMP[(kind, o)] = r
        a.fill_between(ph, r, FIT[(kind, o)], color=COL[o], alpha=.25 if kind == 'A' else .18, lw=0)
        a.plot(ph, FIT[(kind, o)], color=COL[o], lw=3.5, alpha=1 if kind == 'A' else .55,
               label=f'{"vertical" if o == "V" else "horizontal"} rotation, best fit k = {k:.2f}')
    a.axhline(45, color='0.5', lw=1.2, ls=':', zorder=0); a.axhline(90, color='0.5', lw=1.2, ls='--', zorder=0)
    a.set(ylim=[0, 180], yticks=[0, 45, 90, 135, 180], xlabel='Motion phase (deg)', **XT); clean(a)
    a.legend(fontsize=21, frameon=False, loc='upper left'); letter(a, lab, title)
aF = fig.add_subplot(gs[3, 0])
for kind, ls, cl in (('A', '-', 'anisotropic cortex'), ('U', ':', 'isotropic cortex')):
    d = FIT[(kind, 'V')]-FIT[(kind, 'H')]
    aF.fill_between(ph, EMP[(kind, 'V')]-EMP[(kind, 'H')], d, color='k', alpha=.15, lw=0)
    aF.plot(ph, d, ls, color='k', lw=3.5, label=cl)
d = FIT[('A', 'V')]-FIT[('A', 'H')]; i = int(np.argmax(d))
aF.axvline(ph[i], color='0.4', lw=1.5, ls='--', zorder=0)
aF.annotate(f'largest difference\nat {ph[i]:.0f}° ({d[i]:.0f}°)', (ph[i], d[i]), xytext=(ph[i]+25, 6), fontsize=20, va='bottom', arrowprops=dict(arrowstyle='-', color='0.5', lw=1))
aF.set(ylim=[-5, 75], ylabel='Vertical − horizontal best fit (deg)', xlabel='Motion phase (deg)', **XT); clean(aF)
aF.legend(fontsize=21, frameon=False, loc='lower left', bbox_to_anchor=(.02, .07)); letter(aF, 'F', 'Nonrigidity anisotropy')

# G, H: stretch
R = np.load(P.A.data+'heeger_fig6_stretch_k.npy', allow_pickle=True).item(); x = (STRETCHES-1)*100
for c, kind, lab, title in ((1, 'U', 'G', 'Isotropic cortex'), (2, 'A', 'H', 'Anisotropic cortex')):
    a = fig.add_subplot(gs[3, c])
    for o in ('V', 'H'):
        kk = np.array([R[(o, kind, s)] for s in STRETCHES]); sl, ic = np.polyfit(x, kk, 1)   # least-squares line
        a.plot(x, kk, 'o', color=COL[o], ms=11)
        a.plot(x, ic+sl*x, '-', color=COL[o], lw=3.5,
               label=f'{"vertical" if o == "V" else "horizontal"} rotation ({sl*10:+.3f} per 10%)')
    a.set(xlabel='Horizontal image stretch (%)', xticks=[0, 10, 20, 30, 40, 50], ylim=[.3, 1.0], ylabel='Best k (0 rotation, 1 wobble)')
    clean(a); letter(a, lab, title)
    a.legend(frameon=False, loc='lower right', fontsize=21)
fig.savefig(P.A.out+'R2_86_fig6_combined.png', dpi=70)
