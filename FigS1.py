#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Figure S1. Shape matches of the individual observers (as Figure 4B, C): histograms of the horizontal
stretch (image: red; physical: blue; dashed: means) chosen to match the horizontally (left) or vertically (right)
rotating shapes. Data: Toolbox/Data/experiment. Output: figures_paper/FigS1.pdf.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import numpy as np
import figstyle as fs
import experiment_data as E

df, _ = E.load()
fig = fs.figure(200)
gs = fig.add_gridspec(4, 2, hspace=.7, wspace=.25, left=.1, right=.97, top=.95, bottom=.06)
bins = np.arange(1, 1.61, .01)
for oo in range(4):
    d = df[df['ID'] == oo]
    for j, (t, title) in enumerate(((0, 'Matching horizontally rotating shapes'), (1, 'Matching vertically rotating shapes'))):
        ax = fig.add_subplot(gs[oo, j]); im = d['Stretch image adjust'][d['Test'] == t].values; ph = d['Stretch phys adjust'][d['Test'] == t].values
        ax.hist(im, bins, color='r', alpha=.5, hatch='////', ec='r', lw=.5, label='Image stretch')
        ax.hist(ph, bins, color='b', alpha=.5, hatch='\\\\\\\\', ec='b', lw=.5, label='Physical stretch')
        ax.axvline(im.mean(), color='r', ls='--', lw=1, label='Mean (image)'); ax.axvline(ph.mean(), color='b', ls='--', lw=1, label='Mean (physical)')
        ax.set(xlim=[1, 1.6], ylabel='Count', title=title)
        if oo == 3:
            ax.set_xlabel('Horizontal stretch (× width)')
        if j == 0:
            ax.text(-.22, 1.15, f'Observer {oo+1}', transform=ax.transAxes, fontsize=8, fontweight='bold')
        if oo == 0 and j == 1:
            ax.legend(loc='upper left', fontsize=6)
fs.save(fig, 'FigS1')
