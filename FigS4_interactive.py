#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive companion of Supplementary Figure S4 (S4): matching task. Top: the original two rings rotating horizontally and vertically.
Below: horizontal and vertical rings, each with its own knob that goes in 13 steps from the smooth rings (no texture)
to 8-frame random-dot rims whose orientation power moves from concentrated along the rim (16 px streaks) to spread over
all orientations (round dots).
Videos: Toolbox/Data/work/stimfilter_videos (made by Toolbox/revision/Revision_R2_figS_texture_spread_videos.py, which
is run here if any is missing). The page embeds the videos, so it works when opened from any folder.
Output: figures_paper/FigS4_matching_task.html.
"""
import os, sys, shutil, runpy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import figstyle as fs
import Revision_R2_figS_texture_spread_videos as V

if any(not os.path.exists(V.one_path(o, s)) for o in 'HV' for s in ['smooth']+V.PAGE_STREAKS):
    runpy.run_path(V.__file__, run_name='__main__')                 # make the stimulus videos
runpy.run_path(os.path.join(fs.TOOLBOX, 'revision', 'Revision_R2_figS_texture_spread_page.py'), run_name='__main__')
shutil.move(os.path.join(fs.DATA, 'work', 'R2_86_figS_texture_spread.html'), os.path.join(fs.OUT, 'FigS4_matching_task.html'))
print('saved', os.path.join(fs.OUT, 'FigS4_matching_task.html'))
