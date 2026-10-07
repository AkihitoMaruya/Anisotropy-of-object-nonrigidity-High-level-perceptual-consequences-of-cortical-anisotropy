#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common format for all figures (main Figures 1-6 and Supplementary Figures S1-S9).
Width 180 mm (Scientific Reports double column), Arial, 10 pt text (8.75 pt ticks and legends), 12.5 pt bold panel letters,
1.8 pt data lines (sizes below are scaled by FONT_SCALE / LINE_SCALE on saving); panels on a 3-column grid; vertical rotation red, horizontal blue; anisotropic cortex solid,
isotropic dotted (and paler). Saved as PDF (images at 300 dpi) in Codes/figures_paper/."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOOLBOX = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.dirname(TOOLBOX)                                  # works wherever Codes/ is downloaded
OUT = os.path.join(CODES, 'figures_paper')
VOUT = os.path.join(OUT, 'videos')                                # figure videos
DATA = os.path.join(TOOLBOX, 'Data')
os.makedirs(VOUT, exist_ok=True)
for p in (TOOLBOX, os.path.join(TOOLBOX, 'revision')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('K_FIT_TRIM', '10')

MM = 1/25.4
WIDTH = 180*MM                                   # figure width (in)
COL_W = WIDTH/3                                  # one grid column
LW = 1.2                                         # data line width
COL = {'V': '#c0392b', 'H': '#2471a3'}           # vertical / horizontal rotation
ISO_ALPHA = .55                                  # isotropic cortex: dotted, paler
PANEL_FS = 10

STYLE = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False,
    'axes.linewidth': .8, 'xtick.major.width': .8, 'ytick.major.width': .8, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': LW, 'lines.markersize': 3.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.default': 'regular'}


def style():
    """(Re)apply the common style (some model modules change rcParams when imported)."""
    plt.rcParams.update(STYLE)


style()
import logging; logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)   # Arial may be missing (Colab)


def figure(height_mm):
    """A figure 180 mm wide and height_mm tall."""
    return plt.figure(figsize=(WIDTH, height_mm*MM))


def letter(ax, s, x=-.18, y=1.06, title=None):
    """Panel letter (10 pt bold) at the top left of ax, optionally followed by a short title (8 pt)."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=PANEL_FS, fontweight='bold', va='bottom', ha='left')
    if title:
        ax.set_title(title, loc='left', pad=4)


def fig_letter(fig, x, y, s):
    """Panel letter in figure coordinates."""
    fig.text(x, y, s, fontsize=PANEL_FS, fontweight='bold', va='top', ha='left')


FONT_SCALE = 1.25                                # final text = set size x 1.25 (8 pt -> 10 pt)
LINE_SCALE = 1.5                                 # final lines = set width x 1.5


def _enlarge(fig):
    """Scale every text and line of the finished figure (keeps all relative sizes)."""
    import matplotlib.text as mtext, matplotlib.lines as mlines, matplotlib.patches as mpatches, matplotlib.collections as mcoll
    fig.canvas.draw()
    ticks = set()
    for ax in fig.axes:
        for a in (ax.xaxis, ax.yaxis):
            t = a.get_major_ticks()
            if t:
                ticks.update(x for tk in a.get_major_ticks()+a.get_minor_ticks() for x in (tk.label1, tk.label2, tk.tick1line, tk.tick2line))
                a.set_tick_params(which='major', labelsize=t[0].label1.get_fontsize()*FONT_SCALE,
                                  width=t[0].tick1line.get_markeredgewidth()*LINE_SCALE)
    for o in fig.findobj():
        if o in ticks:
            continue
        if isinstance(o, mtext.Text):
            o.set_fontsize(o.get_fontsize()*FONT_SCALE)
        elif isinstance(o, mlines.Line2D):
            o.set_linewidth(o.get_linewidth()*LINE_SCALE); o.set_markeredgewidth(o.get_markeredgewidth()*LINE_SCALE)
        elif isinstance(o, mpatches.Patch):
            o.set_linewidth(o.get_linewidth()*LINE_SCALE)
        elif isinstance(o, mcoll.Collection) and not isinstance(o, mcoll.QuadMesh):
            o.set_linewidths(np.asarray(o.get_linewidths())*LINE_SCALE)


def save(fig, name):
    _enlarge(fig)
    fig.savefig(os.path.join(OUT, f'{name}.pdf'), bbox_inches='tight', pad_inches=.02)   # content spans the full width
    plt.close(fig)
    print('saved', os.path.join(OUT, name+'.pdf'))


def phase_axis(ax, label=True):
    ax.set(xticks=[0, 45, 90, 135, 180], xlim=[0, 180])
    if label:
        ax.set_xlabel('Motion phase (deg)')


VIDEOS = os.path.join(DATA, 'videos')


def video_frame(path, frac=0.0):
    """One frame (fraction frac of the duration) of a video."""
    import imageio.v2 as imageio
    r = imageio.get_reader(path); n = r.count_frames(); return r.get_data(min(int(n*frac), n-1))


def play_badge(fig, ax, size_mm=5):
    """Small play mark at the lower right of an image axis (the panel is a video)."""
    import imageio.v2 as imageio
    p = ax.get_position(); W, H = fig.get_size_inches(); w, h = size_mm*MM/W, size_mm*MM/H
    a = fig.add_axes([p.x1-w*1.1, p.y0+h*.1, w, h]); a.imshow(imageio.imread(os.path.join(VIDEOS, 'playmark.png'))); a.axis('off')


def combine_videos(tiles, name, size, fps=30, duration=None):
    """One video from several, each in its own box, playing at its own speed; shorter ones loop.
    tiles: list of dict(src=path to an mp4 or a list of frames, fps=frames per second of a frame list,
    box=(x, y, w, h) in pixels, label=panel letter or ''). size = (W, H) of the output (even numbers).
    duration: seconds (default: the longest tile). Saved in figures_paper/videos/."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    class Source:
        def __init__(self, t):
            self.t, self.frames = t, t['src'] if not isinstance(t['src'], str) else None
            if self.frames is None:
                r = imageio.get_reader(t['src']); m = r.get_meta_data()
                self.fps, self.n = m['fps'], r.count_frames(); r.close()
            else:
                self.fps, self.n = t['fps'], len(self.frames)
            self.reader, self.pos, self.cur = None, -1, None

        def get(self, sec):                                          # frame shown at time sec (looping)
            i = int(sec*self.fps) % self.n
            if self.frames is not None:
                return self.frames[i]
            if self.reader is None or i < self.pos:                  # (re)start reading from the beginning
                if self.reader is not None:
                    self.reader.close()
                self.reader, self.pos = imageio.get_reader(self.t['src']), -1
                self.it = self.reader.iter_data()
            while self.pos < i:
                self.cur = next(self.it); self.pos += 1
            return self.cur

    S = [Source(t) for t in tiles]
    T = duration or max(s.n/s.fps for s in S)
    try:
        font = ImageFont.truetype('Arial Bold.ttf', 30)
    except OSError:
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 30)
        except OSError:
            font = ImageFont.load_default(size=30)
    w = imageio.get_writer(os.path.join(VOUT, name), fps=fps, codec='libx264', quality=7, macro_block_size=1)
    for k in range(int(round(T*fps))):
        canvas = Image.new('RGB', size, 'white'); d = ImageDraw.Draw(canvas)
        for s in S:
            x, y, bw, bh = s.t['box']
            im = Image.fromarray(np.asarray(s.get(k/fps))[..., :3]).resize((bw, bh), Image.LANCZOS)
            canvas.paste(im, (x, y))
            if s.t.get('label'):
                d.text((x-34 if x >= 34 else x+4, y), s.t['label'], fill='black', font=font)
        w.append_data(np.asarray(canvas))
    w.close(); print('saved', os.path.join(VOUT, name))


def copy_video(src, name):
    """Copy a video into figures_paper/videos/ under the figure's name (e.g. 'Fig1C.mp4')."""
    import shutil
    shutil.copy(src, os.path.join(VOUT, name)); print('saved', os.path.join(VOUT, name))


def image_panel(ax, img):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
