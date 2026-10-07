#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4. Effect of image and physical stretch on perceived nonrigidity.
A: rings stretched horizontally in the image (rows 1, 3) or physically before projection (rows 2, 4), from 0% to 50%
   (geometry of the experiment's stimuli, as in the original Figure4_A.py; first frame). The experiment's video:
   figures_paper/videos/Fig4A.mp4.
B, C: histograms of the horizontal stretch (image: red; physical: blue) chosen to match the shapes, when the
   horizontally (B) or vertically (C) rotating rings were the test shape.
D: proportion of trials on which the vertically rotating rings were judged more nonrigid (mean +- s.e. of bootstrap
   means; * p < .05, one-sample t-test against 0.5), conditions as in the original Figure4_B-D.py.
Data: Toolbox/Data/experiment/. Output: figures_paper/Fig4.pdf; videos/Fig4A.mp4.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import numpy as np
from scipy.stats import ttest_1samp
import figstyle as fs

IMG, PHYS = 'r', 'b'                                           # image stretch red, physical stretch blue (as the original Figure 4)


# ---------------------------------------------------------------- A: stimuli
def rings(a, b, w, h, im_rot, Om=0.0, phi=np.radians(30), dc=100.):
    th = np.arange(0, 2*np.pi, .01); out = []
    for s in (1, -1):
        X = a*np.cos(Om)*np.cos(th)-b*np.sin(Om)*np.sin(th)*np.cos(s*phi)
        Y = b*np.sin(th)*np.sin(s*phi)-s*np.sin(phi)*b
        Z = -a*np.sin(Om)*np.cos(th)-b*np.cos(Om)*np.sin(th)*np.cos(s*phi)
        u, v = X*dc/(dc-Z), Y*dc/(dc-Z)
        ur, vr = u*np.cos(im_rot)-v*np.sin(im_rot), u*np.sin(im_rot)+v*np.cos(im_rot)
        out.append((w*ur, h*vr))
    return out


def stretched(kind, rot, A):
    """kind: 'image' or 'physical'; rot: 0 (horizontal) or 90 (vertical); returns ring outlines (as Figure4_A.py)."""
    a1 = b1 = w1 = h1 = 1.
    if kind == 'image':
        w1, h1 = (A, 1.) if rot == 0 else (1., A)
    else:
        a1, b1 = (A, 1.) if rot == 0 else (1., A)
    w, h = w1/np.sqrt(w1*h1), h1/np.sqrt(w1*h1); a, b = a1/np.sqrt(a1*b1), b1/np.sqrt(a1*b1)
    return rings(a, b, w, h, np.radians(rot))


# ---------------------------------------------------------------- B-D: data (as Figure4_B-D.py)
import experiment_data as E
df, Max_res = E.load(); Test = df['Test']
rng = np.random.default_rng(0)
B = {k: E.bootstrap(v, rng) for k, v in E.conditions(df, Max_res).items()}

# ---------------------------------------------------------------- figure
fig = fs.figure(120)
lev = [1.0, 7/6, 4/3, 1.5]
rows = (('image', 0), ('physical', 0), ('image', 90), ('physical', 90))
x0, y0, cw, ch = .13, .05, .095, .19
bg = fig.add_axes([x0-.005, y0-.005, 4*cw+.005, 4*ch+.005]); bg.set_facecolor('0.5'); bg.set_xticks([]); bg.set_yticks([])
for sp in bg.spines.values():
    sp.set_visible(False)
for r, (kind, rot) in enumerate(rows):
    for c, A in enumerate(lev):
        ax = fig.add_axes([x0+c*cw, y0+(3-r)*ch, cw*.95, ch*.95])
        for u, v in stretched(kind, rot, A):
            ax.plot(np.r_[u, u[0]], np.r_[v, v[0]], 'w', lw=1.2)
        ax.set(xlim=[-1.6, 1.6], ylim=[-1.6, 1.6], aspect='equal'); ax.axis('off')
    fig.text(x0-.01, y0+(3-r)*ch+ch*.47, f'{kind.capitalize()} stretch', ha='right', va='center', fontsize=7,
             color=IMG if kind == 'image' else PHYS)
fig.text(x0, y0+4*ch+.012, '0% stretch', fontsize=7, va='bottom'); fig.text(x0+4*cw, y0+4*ch+.012, '50% stretch', fontsize=7, ha='right', va='bottom')
fig.patches.append(__import__('matplotlib').patches.FancyArrow(x0+.02, y0+4*ch+.06, 4*cw-.05, 0, width=.008, head_width=.022, head_length=.015,
                   transform=fig.transFigure, color='#4472c4'))
fig.text(x0+2*cw, y0+4*ch+.08, 'Horizontal stretch', ha='center', fontsize=8)
fs.play_badge(fig, bg)
fs.fig_letter(fig, .005, .99, 'A')
for j, (t, title, lab) in enumerate(((0, 'Matching horizontally rotating shapes', 'B'), (1, 'Matching vertically rotating shapes', 'C'))):
    ax = fig.add_axes([.6, .75-j*.29, .38, .19])
    im_, ph_ = df['Stretch image adjust'][Test == t].values, df['Stretch phys adjust'][Test == t].values
    bins = np.linspace(1, 1.6, 21)
    ax.hist(im_, bins, color=IMG, alpha=.5, hatch='////', ec=IMG, lw=.5, label='Image stretch')
    ax.hist(ph_, bins, color=PHYS, alpha=.5, hatch='\\\\\\\\', ec=PHYS, lw=.5, label='Physical stretch')
    ym = 1/2.3 if j == 1 else 1                              # C: mean lines stop below the legend
    ax.axvline(im_.mean(), ymax=ym, color=IMG, ls='--', lw=1, label='Mean (image)'); ax.axvline(ph_.mean(), ymax=ym, color=PHYS, ls='--', lw=1, label='Mean (physical)')
    ax.set(xlim=[1, 1.6], ylabel='Count', title=title)
    ax.set_xlabel('Horizontal stretch (× width)') if j == 1 else ax.set_xticklabels([])
    if j == 1:
        ax.set_ylim(0, ax.get_ylim()[1]*2.3)                 # room for the legend above the bars
        ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1.04), ncol=2, fontsize=6, columnspacing=1)
    fs.letter(ax, lab, x=-.2)
ax = fig.add_axes([.6, .07, .38, .25])
labels = list(B); m = [B[k].mean() for k in labels]; se = [B[k].std(ddof=1)/np.sqrt(1000) for k in labels]
p = {k: ttest_1samp(B[k], .5).pvalue for k in labels}
cols = ['k', IMG, PHYS, IMG, PHYS, IMG]
ax.bar(range(6), m, .6, color=cols, yerr=se, capsize=2, error_kw=dict(lw=.8))
for i, k in enumerate(labels):
    if p[k] < .05:
        ax.text(i, m[i]+.02, '*', ha='center', fontsize=9)
ax.axhline(.5, color='0.5', lw=.8, ls=':')
ax.set(xticks=range(6), xticklabels=labels, yticks=[0, .5, 1], ylim=[0, 1.1], ylabel='Proportion vertical\nmore nonrigid')
fs.letter(ax, 'D', x=-.2)
fs.copy_video(os.path.join(fs.VIDEOS, 'Fig4A.mp4'), 'Fig4A.mp4')
fs.save(fig, 'Fig4')
print({k: round(v, 3) for k, v in zip(labels, m)}, {k: f'{v:.2g}' for k, v in p.items()})
