#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1. Anisotropy of a rotating ring illusion (videos).
A: two Styrofoam rings glued together at an angle rotate together on a turntable. B: rotated by 90 deg, one ring
appears to wobble against the other. C, D: circular (C) and octagonal (D) rings physically rotating together.
E, F: the two rings physically wobbling against each other. G, H: C and D rotated by 90 deg.
Panels are the first frames of the videos (Toolbox/Data/videos; C-H made with Toolbox/Make_rotating_two_rings.py);
the videos are copied to figures_paper/videos/Fig1A.mp4 ... Fig1H.mp4.
Output: figures_paper/Fig1.pdf; videos/Fig1A-H.mp4.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import figstyle as fs

fig = fs.figure(96)
w, h = .235, .44
for i, p in enumerate('ABCDEFGH'):
    src = os.path.join(fs.VIDEOS, f'Fig1{p}.mp4')
    ax = fig.add_axes([.015+(i % 4)*.247, .52-(i//4)*.49, w, h])
    fs.image_panel(ax, fs.video_frame(src)); fs.play_badge(fig, ax)
    fs.letter(ax, p, x=-.04, y=1.01)
    fs.copy_video(src, f'Fig1{p}.mp4')
fs.save(fig, 'Fig1')
