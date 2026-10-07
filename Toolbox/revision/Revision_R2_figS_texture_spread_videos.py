#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-ring stimulus (Revision_R2_figS_two_ring_filter_videos.py), horizontal and vertical rotation, with random-dot rims whose elements live LIFE frames (about 270 ms) and then reappear at a
random new place (staggered), so no element can be tracked through the rotation. Knob: the orientation power of the texture, from
concentrated along the rim (every dot stretched along the ring, Gaussian arc SD 16 px) to scattered over all
orientations (round dots). For the two-knob matching page, one mp4 per rotation and streak length (the page's finer
set, PAGE_STREAKS): Toolbox/Data/work/stimfilter_videos/tex_{H|V}_s{streak}.mp4, and the smooth rings smooth_{H|V}.mp4.
New analysis."""
import os, sys
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import numpy as np
import imageio.v2 as imageio
from Revision_R2_figS_two_ring_filter_videos import two_ring_texture_video, two_ring_video, to8
OUT_V = os.path.join(current_folder, 'Toolbox/Data/work/stimfilter_videos/')
STREAKS = [16.0, 8.0, 4.0, 2.0, 1.0, 0.0]                         # supplementary figure / movie
PAGE_STREAKS = [16.0, 11.3, 8.0, 5.7, 4.0, 2.8, 2.0, 1.4, 1.0, 0.7, 0.5, 0.0]   # matching page: steps of about sqrt(2)
LIFE = 8


def one_path(o, st):
    """st: streak length, or 'smooth' for the original rings without texture."""
    return OUT_V+(f'smooth_{o}.mp4' if st == 'smooth' else f'tex_{o}_s{st:g}.mp4')


def save_one(path, v):
    """One rotation alone (same grey scale and 2x up-sampling as save())."""
    s = np.percentile(np.abs(v), 99.5)
    fr = np.repeat(np.repeat(to8(v, -s, s), 2, 1), 2, 2)
    imageio.mimsave(path, list(fr), fps=30, codec='libx264', quality=8, macro_block_size=1)


def make_single():
    """Per-rotation videos for the two-knob page."""
    for o in ('H', 'V'):
        if not os.path.exists(one_path(o, 'smooth')):
            save_one(one_path(o, 'smooth'), two_ring_video(o))
        for st in PAGE_STREAKS:
            if not os.path.exists(one_path(o, st)):
                save_one(one_path(o, st), two_ring_texture_video(o, streak=st, lifetime=LIFE)); print(o, st, flush=True)

if __name__ == '__main__':
    make_single()
