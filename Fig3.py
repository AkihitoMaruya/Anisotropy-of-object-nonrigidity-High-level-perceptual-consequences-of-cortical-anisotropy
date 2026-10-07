#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3. Cortical anisotropy (computations as in the original Figure3.py).
A: orientation tuning curves of V1 simple cells (von Mises), weighted by cell number, anisotropic and isotropic cortex.
B: cell numbers and tuning widths by preferred orientation (anisotropic: red; isotropic: grey dotted), and decoded
angles around the horizontal (gamma_H) and vertical (gamma_V) axes against the physical angle.
C: decoded angle difference gamma_H - gamma_V.
Data: Toolbox/Data/experiment/d.pt (cell numbers) and K1s.pt (von Mises concentrations).
Output: figures_paper/Fig3.pdf.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import numpy as np
import torch
from scipy.stats import vonmises
import matplotlib.pyplot as plt
import figstyle as fs

D = os.path.join(fs.DATA, 'experiment')
Ns = (torch.load(D+'/d.pt')[:8].reshape(-1)+torch.load(D+'/d.pt')[8:16].reshape(-1))/2
K1s = (torch.load(D+'/K1s.pt')[:8].reshape(-1)+torch.load(D+'/K1s.pt')[8:16].reshape(-1))/2
Ns, K1s = torch.hstack((Ns, Ns)), torch.hstack((K1s, K1s))
pfo = torch.arange(-np.pi, np.pi, np.pi/8)
TW = np.array([28.138, 29.793, 34.759, 38.069, 35.586, 40.552, 33.931, 30.207,
               28.552, 32.276, 34.759, 38.483, 35.586, 37.655, 33.931, 32.276])   # tuning widths (deg)
TW = np.hstack(((TW[:8]+TW[8:])/2,)*2)


def get_response(pf, thetac, K, thetas, k=0.125):
    lc, ls = vonmises.pdf(pf, K, thetac), vonmises.pdf(pf, K, thetas)
    return lc/np.sqrt(lc**2+ls**2+k)


def angle_estimate(Ns, pfo, K1s, thetac, thetas, k=0.125):
    v1, v2 = np.zeros(2), np.zeros(2)
    for i in range(len(pfo)):
        n = Ns[i].numpy(); u = np.array([np.cos(pfo[i]), np.sin(pfo[i])])
        v1 += u*n*get_response(pfo[i], thetac, K1s[i], thetas, k)*n
        v2 += u*n*get_response(pfo[i], thetas, K1s[i], thetac, k)
    a1, a2 = np.degrees(np.arctan2(v1[1], v1[0])), np.degrees(np.arctan2(v2[1], v2[0]))
    return abs(a1-a2)


def decode(N, K):
    th = np.arange(5, 90-6, 1); phys, gH, gV = [], [], []
    for c in th:
        tcV, tcH = 90-c, c
        gV.append(angle_estimate(N, pfo, K, np.radians(tcH), np.radians(180-c)))   # as in Figure3.py (H/V names swapped there)
        gH.append(angle_estimate(N, pfo, K, np.radians(tcV), np.radians(-tcV)))
        phys.append(abs(tcH-(180-c)))
    return np.array(phys), np.array(gH), np.array(gV)


physA, gHA, gVA = decode(Ns, K1s)
physU, gHU, gVU = decode(torch.ones_like(Ns), torch.ones_like(K1s))
STY = {'A': dict(color='#d62728', ls='-', label='Anisotropic'), 'U': dict(color='0.5', ls=':', label='Isotropic')}

fig = fs.figure(105)
gs = fig.add_gridspec(2, 12, hspace=.8, wspace=7, left=.08, right=.98, top=.92, bottom=.1)
gtop = gs[0, :].subgridspec(1, 3, wspace=.5)
x = np.linspace(-np.pi, np.pi, 2000); cols = plt.cm.rainbow(np.linspace(0, 1, len(pfo)))
for j, (kind, title, ylab) in enumerate((('A', 'Anisotropic V1', r'$n_i f(\theta|\mu_i,k_i)$'),
                                         ('U', 'Isotropic V1', r'$\bar{n} f(\theta|\mu_i,\bar{k})$'))):
    ax = fig.add_subplot(gtop[j])
    for i in range(len(pfo)):
        k_, n_ = (K1s[i], Ns[i]) if kind == 'A' else (K1s.mean(), Ns.mean())
        ax.plot(np.degrees(x), vonmises.pdf(pfo[i], k_, x)*n_.numpy(), color=cols[i], lw=.9)
    ax.set(xticks=np.arange(-180, 181, 90), xlim=[-180, 180], xlabel='Stimulus orientation (deg)', ylabel=ylab, title=title)

ax = fig.add_subplot(gtop[2])
ax.plot(physA, gHA-gVA, **STY['A']); ax.plot(physU, gHU-gVU, **STY['U'])
ax.set(xticks=np.arange(0, 181, 45), xlabel='Physical angle (deg)', ylabel=r'$\hat{\gamma}_H-\hat{\gamma}_V$ (deg)'); ax.legend(loc='lower right')
po = np.arange(-180, 180, 22.5)
ax = fig.add_subplot(gs[1, 0:3])
ax.plot(po, Ns, marker='o', **STY['A']); ax.plot(po, Ns.mean()*np.ones(16), marker='o', **STY['U'])
ax.set(xticks=np.arange(-180, 181, 90), ylim=[200, 500], xlabel='Preferred orientation (deg)', ylabel='Number of cells')
ax = fig.add_subplot(gs[1, 3:6])
ax.plot(po, TW, marker='o', **STY['A']); ax.plot(po, TW.mean()*np.ones(16), marker='o', **STY['U'])
ax.set(xticks=np.arange(-180, 181, 90), ylim=[20, 45], xlabel='Preferred orientation (deg)', ylabel='Tuning width (deg)')
for j, (gA, gU, lab) in enumerate(((gHA, gHU, r'$\hat{\gamma}_H$ (deg)'), (gVA, gVU, r'$\hat{\gamma}_V$ (deg)'))):
    ax = fig.add_subplot(gs[1, 6+3*j:9+3*j])
    ax.plot(physA, gA, **STY['A']); ax.plot(physU, gU, **STY['U'])
    ax.set(xticks=np.arange(0, 181, 90), yticks=np.arange(0, 181, 90), xlim=[0, 180], ylim=[0, 180], aspect='equal',
           xlabel='Physical angle (deg)', ylabel=lab)
    if j == 0:
        ax.legend(loc='lower right', bbox_to_anchor=(1.06, -.03), fontsize=6, handlelength=1.6)
fs.fig_letter(fig, .005, .99, 'A'); fs.fig_letter(fig, .69, .99, 'C'); fs.fig_letter(fig, .005, .47, 'B')
fs.save(fig, 'Fig3')
