#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Figure S2. Nonrigidity judgements of the individual observers (as Figure 4D): proportion of trials on
which the vertically rotating rings were judged more nonrigid (mean +- s.e. of bootstrap means; * p < .05, one-sample
t-test against 0.5). Max: each observer's own Max-condition trials. Data: Toolbox/Data/experiment.
Output: figures_paper/FigS2.pdf.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import numpy as np
from scipy.stats import ttest_1samp
import figstyle as fs
import experiment_data as E

df, max_res = E.load(); rng = np.random.default_rng(0)
fig = fs.figure(110)
gs = fig.add_gridspec(2, 2, hspace=.5, wspace=.3, left=.1, right=.97, top=.93, bottom=.08)
cols = ['k', 'r', 'b', 'r', 'b', 'r']
for oo in range(4):
    C = E.conditions(df[df['ID'] == oo], [max_res[oo]]); B = {k: E.bootstrap(v, rng) for k, v in C.items()}
    ax = fig.add_subplot(gs[oo//2, oo % 2]); labels = list(B)
    m = [B[k].mean() for k in labels]; se = [B[k].std(ddof=1)/np.sqrt(1000) for k in labels]
    ax.bar(range(6), m, .6, color=cols, yerr=se, capsize=2, error_kw=dict(lw=.8))
    for i, k in enumerate(labels):
        if ttest_1samp(B[k], .5).pvalue < .05:
            ax.text(i, m[i]+.02, '*', ha='center', fontsize=9)
    ax.axhline(.5, color='0.5', lw=.8, ls=':')
    ax.set(xticks=range(6), xticklabels=labels, yticks=[0, .5, 1], ylim=[0, 1.12], title=f'Observer {oo+1}')
    if oo % 2 == 0:
        ax.set_ylabel('Proportion vertical\nmore nonrigid')
fs.save(fig, 'FigS2')
