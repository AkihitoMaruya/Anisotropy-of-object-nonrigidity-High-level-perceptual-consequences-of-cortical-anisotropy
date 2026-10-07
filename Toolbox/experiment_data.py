#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Psychophysical data (Toolbox/Data/experiment), loaded as in the original Figure4_B-D.py / FigureS1-2.py:
shape-match settings (mean of three settings per trial, as stretch factors 1-1.5) and nonrigidity choices.
load() -> (df, max_res): df one row per trial (column ID = observer 0-3), max_res list of the Max-condition choices."""
import os, pickle
import numpy as np
import pandas as pd
import figstyle as fs

D = os.path.join(fs.DATA, 'experiment')+'/'
NAMES = ['Obs1', 'Obs2', 'Obs3', 'Obs4']                    # observers 1-4 (anonymised)
INTERVAL = [[(0, 59)], [(0, 27), (27, 39)], [(0, 45)], [(0, 18), (18, 39)]]
STRETCH = np.linspace(1, 1.5, 12)


def load():
    frames, max_res = [], []
    for nn, name in enumerate(NAMES):
        max_res.append(np.load(D+name+'_data.npy'))
        for a0, a1 in INTERVAL[nn]:
            with open(D+name+f'/{name}_{a0}_to_{a1}_data.pkl', 'rb') as f:
                d = pickle.load(f)
            im = [STRETCH[d['Stretch image adjust'][i:i+3].astype(int)] for i in range(0, len(d['Stretch image adjust']), 3)]
            ph = [STRETCH[d['Stretch phys adjust'][i:i+3].astype(int)] for i in range(0, len(d['Stretch phys adjust']), 3)]
            d['Stretch image adjust'] = np.mean(np.vstack(im), 1); d['Stretch phys adjust'] = np.mean(np.vstack(ph), 1)
            d['Pattern'] = d['Pattern'][:len(d['Stretch image adjust'])]; d['Test'] = d['Test'][:len(d['Stretch image adjust'])]
            pre = pd.DataFrame(d)
            pre = pre.iloc[:40] if len(pre) > 40 else pre
            pre['ID'] = nn
            frames.append(pre)
    return pd.concat(frames), max_res


def conditions(df, max_res):
    """Choices per condition (1 = vertical rotation judged more nonrigid)."""
    T = df['Test']
    return {'Original': df['H VS V'].values, '$H_i$': 1-df[T == 1]['Non-rigidity image'].values,
            '$H_p$': 1-df[T == 1]['Non-rigidity phys'].values, '$V_i$': df['Non-rigidity image'][T == 0].values,
            '$V_p$': df['Non-rigidity phys'][T == 0].values, 'Max': np.hstack(max_res)}


def bootstrap(x, rng, n=1000):
    return np.array([rng.choice(x, 1000, replace=True).mean() for _ in range(n)])
