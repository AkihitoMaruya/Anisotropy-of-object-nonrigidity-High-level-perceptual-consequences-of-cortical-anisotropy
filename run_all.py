#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make every figure of the paper (Figures 1-6, Supplementary Figures S1-S5) into figures_paper/.
Usage: python run_all.py            (all)
       python run_all.py Fig5 FigS7 (selected)
Works from any location: all paths are relative to this folder. Model results are cached in Toolbox/Data/Rings
(missing ones are recomputed by the model code in Toolbox/revision, which takes long)."""
import os, sys, runpy, time
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = ['Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6'] + ['FigS1', 'FigS2', 'FigS3', 'FigS3_filters_interactive', 'FigS4', 'FigS4_interactive', 'FigS5', 'FigS6', 'FigS7']
LOG = os.environ.get('LOG_OPENS')
if LOG:                                                             # optional: record every data file read
    seen = set()
    def hook(ev, args):
        if ev == 'open' and isinstance(args[0], str) and 'Toolbox/Data' in args[0] and args[0] not in seen:
            seen.add(args[0]); open(LOG, 'a').write(args[0]+'\n')
    sys.addaudithook(hook)
todo = [f for f in FIGS if len(sys.argv) == 1 or any(f.startswith(a) for a in sys.argv[1:])]
for f in todo:
    t0 = time.time(); print(f'--- {f}', flush=True)
    sys.argv = [f+'.py']
    runpy.run_path(os.path.join(HERE, f+'.py'), run_name='__main__')
    import matplotlib.pyplot as plt; plt.close('all')
    print(f'    {time.time()-t0:.0f} s', flush=True)
