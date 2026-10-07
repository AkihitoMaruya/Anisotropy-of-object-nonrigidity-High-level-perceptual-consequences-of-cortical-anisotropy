#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary movie: the experimental two-ring stimulus (Revision_R2_figS_two_ring_filter_videos.py) rotating
vertically (or, with the argument H, horizontally), side by side: smooth rings (original) and random-dot rims with limited lifetime (8 frames, then redrawn
at a random new place) whose dots are stretched along the rim into streaks (Gaussian arc SD 16, 8, 4, 2, 1 px) or
round, i.e. orientation energy from concentrated along the rim to spread over all orientations. Each panel's
contrast symmetric about mid-grey. Outputs: Toolbox/Data/work/R2_86_figS_two_ring_streaks.mp4 and a still (.png);
horizontal rotation: R2_86_figS_two_ring_streaks_H.mp4 / .png.
New analysis."""
import os, sys
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from Revision_R2_figS_two_ring_filter_videos import two_ring_video, two_ring_texture_video, to8
from Revision_R2_figS_texture_spread_videos import STREAKS, LIFE
OUT = current_folder+'/Toolbox/Data/work/'
try:
    FONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 22)
except OSError:                                                      # not macOS (e.g. Colab)
    FONT = ImageFont.load_default(size=22)
O = 'H' if 'H' in sys.argv[1:] else 'V'
NAME = OUT+'R2_86_figS_two_ring_streaks'+('_H' if O == 'H' else '')

if __name__ == '__main__':
    vids = [('smooth rings', two_ring_video(O))]
    vids += [(f'streaks {s:g} px' if s else 'round dots', two_ring_texture_video(O, streak=s, lifetime=LIFE)) for s in STREAKS]
    act = np.max([np.abs(v).max(0) for _, v in vids], 0) > 1e-3        # crop to where any stimulus has contrast
    rows, cols = np.where(act.any(1))[0], np.where(act.any(0))[0]
    r0, r1, c0, c1 = max(rows[0]-6, 0), min(rows[-1]+7, act.shape[0]), max(cols[0]-6, 0), min(cols[-1]+7, act.shape[1])
    pan = [to8(v[:, r0:r1, c0:c1], -np.percentile(np.abs(v), 99.5), np.percentile(np.abs(v), 99.5)) for _, v in vids]
    h, w, gap, top = r1-r0, c1-c0, 8, 40                               # native resolution
    W = len(pan)*(w+gap)-gap; W += W % 2; H = top+h; H += H % 2
    lab = Image.new('L', (W, top), 255); d = ImageDraw.Draw(lab)
    for i, (name, _) in enumerate(vids):
        d.text((i*(w+gap)+w//2, top//2), name, font=FONT, fill=0, anchor='mm')
    lab = np.asarray(lab)
    frames = []
    for t in range(len(pan[0])):
        f = np.full((H, W), 255, np.uint8); f[:top] = lab
        for i, p in enumerate(pan):
            f[top:top+h, i*(w+gap):i*(w+gap)+w] = p[t]
        frames.append(f)
    imageio.mimsave(NAME+'.mp4', frames, fps=30, codec='libx264', quality=8, macro_block_size=1)
    imageio.imwrite(NAME+'.png', frames[len(frames)//8])
    print(W, H)
