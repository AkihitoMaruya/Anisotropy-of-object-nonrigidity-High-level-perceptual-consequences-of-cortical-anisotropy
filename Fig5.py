#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5. Optic flows from isotropic and anisotropic cortex (revised model, Toolbox/revision/fig5_model.py).
A, B: direction tuning of the motion-energy units (grating drifting at 0.58 px/frame), times the cell number
      (A anisotropic: cat widths x2.5 and cat numbers; B isotropic: mean width and number). Colour = preferred direction.
C: optic flow on the rotating ring at 45 deg phase (columns horizontal / vertical rotation; rows isotropic / anisotropic),
   colour = direction; the flow over the cycle is figures_paper/videos/Fig5C.mp4.
D, E: the original Figure 5D and 5E videos (snapshots; figures_paper/videos/Fig5D.mp4, Fig5E.mp4).
F, G: cosine similarity between the flow and rotation-to-wobble template directions against k (mean +- SD/2), best k.
Output: figures_paper/Fig5.pdf; videos/Fig5C-E.mp4.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import numpy as np
import imageio.v2 as imageio
import figstyle as fs
import fig5_model as M
fs.style()

alpha = np.radians(np.arange(-180, 180, 1.0)); PHI_S = np.radians(30)
fg = M.f0*np.stack([np.cos(PHI_S)*np.cos(alpha), np.cos(PHI_S)*np.sin(alpha), -np.full_like(alpha, np.sin(PHI_S))], 1)
VIDS = {o: M.d4g_ring_video(o, sigma=1.5) for o in ('H', 'V')}
SEL = M.K <= 1.0
BK = {}


def draw_tuning(ax, kind):
    b = M.bank(kind); G = np.zeros((len(b.mu), len(alpha)))
    for i, (mu, C) in enumerate(zip(b.mu, b.C)):
        G[i] = np.exp(-np.sum((fg-mu)**2, 1)/C[0, 0])+np.exp(-np.sum((-fg-mu)**2, 1)/C[0, 0])
    R = G.reshape(16, b.S, -1)[:, M.LAYOUT_C < 0].sum(1); R = R/R.max(1, keepdims=True)
    n = M.NUM if kind == 'A' else np.full(16, M.NUM.mean())
    for k in range(16):
        pk = alpha[np.argmax(R[k])]
        ax.plot(np.degrees(alpha), n[k]*R[k], color=M.dir_rgb(np.array([np.cos(pk), np.sin(pk)])), lw=1)
    ax.set(xticks=np.arange(-180, 181, 90), xlim=[-180, 180], ylim=[0, 1.4], xlabel='Motion direction (deg)',
           ylabel=r'$n_i m_i$' if kind == 'A' else r'$\bar{n} m_i$', title='ME anisotropy' if kind == 'A' else 'ME isotropy')


def draw_key(ax):
    ang = np.radians(np.arange(0, 360, 45)); v = np.stack([np.cos(ang), np.sin(ang)], 1)
    ax.quiver(np.zeros(8), np.zeros(8), v[:, 0], v[:, 1], color=M.dir_rgb(v), angles='xy', scale_units='xy', scale=1,
              width=.04, headwidth=3.5, headlength=4)
    for (x, y), t in zip(v[::2]*1.6, ('right', 'up', 'left', 'down')):
        ax.text(x, y, t, ha='center', va='center', fontsize=6)
    ax.set(xlim=[-2.2, 2.2], ylim=[-2.2, 2.2], aspect='equal'); ax.axis('off')


def draw_flow(ax, o, kind, fi, title=True):
    V = M.F[(o, kind)][fi]; Pp = M.LP[o][fi]; u = V/np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9); s = slice(None, None, 6)
    ax.imshow(VIDS[o][M.FRAMES[fi]], cmap='gray', vmin=-1.5, vmax=1.5, alpha=.5)
    ax.quiver(Pp[s, 1], Pp[s, 0], u[s, 0], -u[s, 1], color=M.dir_rgb(V[s]), angles='xy', scale_units='xy', scale=.025,
              width=.012, headwidth=3.2, headlength=3.8)
    if title:
        cortex = f'{"isotropic" if kind == "U" else "anisotropic"} cortex'            # rotation named once, above the top row
        ax.set_title(f'{"Horizontal" if o == "H" else "Vertical"} rotation\n{cortex}' if kind == 'U' else cortex, fontsize=7)
    ax.axis('off')


def flow_grid(fig, rect, fi):
    x0, y0, w, h = rect
    for c, o in enumerate(('H', 'V')):
        for r, kind in enumerate(('U', 'A')):
            draw_flow(fig.add_axes([x0+c*w/2, y0+(1-r)*h/2, w/2*.98, h/2*.82]), o, kind, fi)


def draw_cos(ax, kind):
    for o, lw, z in (('V', 2.4, 1), ('H', 1.2, 2)):                   # horizontal on top (they coincide when isotropic)
        m, sd = M.COS[(kind, o)]; m, sd, k = m[SEL], sd[SEL], M.K[SEL]; kb = k[np.argmax(m)]
        ax.plot(k, m, color=fs.COL[o], lw=lw, zorder=z, label=f'{"vertical" if o == "V" else "horizontal"} rotation')
        ax.fill_between(k, m-sd/2, m+sd/2, color=fs.COL[o], alpha=.15, lw=0)
        ax.axvline(kb, ls='--', color=fs.COL[o], lw=.9)
        BK.setdefault(kind, []).append((o, kb))
    ax.set(xticks=np.arange(0, 1.1, .2), yticks=[0, .5, 1], ylim=[0, 1.6], xlabel=r'$k$', ylabel='Cosine similarity\nto template',
           title='Isotropic cortex' if kind == 'U' else 'Anisotropic cortex')
    ar = dict(arrowstyle='-|>', lw=.9, color='k', mutation_scale=7)
    ax.annotate('', xy=(0., -.36), xytext=(.36, -.36), xycoords='axes fraction', arrowprops=ar, annotation_clip=False)
    ax.annotate('', xy=(1., -.36), xytext=(.64, -.36), xycoords='axes fraction', arrowprops=ar, annotation_clip=False)
    ax.text(.18, -.34, 'Rotation', transform=ax.transAxes, ha='center', va='bottom', fontsize=7)
    ax.text(.82, -.34, 'Wobble', transform=ax.transAxes, ha='center', va='bottom', fontsize=7)
    (oa, ka), (ob, kb2) = BK[kind]
    txt = f'best k = {ka:.2f}' if abs(ka-kb2) < .005 else None
    for i, (o_, k_) in enumerate(BK[kind]):
        if txt and i:
            break
        ax.text(k_+.02, 1.06+.0*i, txt or f'{k_:.2f}', color='k' if txt else fs.COL[o_], fontsize=7, ha='left' if (txt or o_ == 'V') else 'right',
                va='bottom', transform=ax.transData) if not (not txt and o_ == 'H') else ax.text(k_-.02, 1.06, f'{k_:.2f}', color=fs.COL[o_], fontsize=7, ha='right', va='bottom')
    if kind == 'U':
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.04))


if __name__ == '__main__':
    H = 252.; fig = fs.figure(H); y = lambda mm: 1-mm/H; hh = lambda mm: mm/H
    axA = fig.add_axes([.07, y(42), .36, hh(34)]); draw_tuning(axA, 'A'); fs.letter(axA, 'A', x=-.2)
    axB = fig.add_axes([.53, y(42), .36, hh(34)]); draw_tuning(axB, 'U'); fs.letter(axB, 'B', x=-.2)
    draw_key(fig.add_axes([.9, y(37), .1, hh(26)]))
    flow_grid(fig, [.0, y(136), .5, hh(76)], M.FI45); fs.fig_letter(fig, .005, y(57), 'C')
    a = fig.add_axes([.0, y(136), .5, hh(76)]); a.axis('off'); fs.play_badge(fig, a)
    rD = imageio.get_reader(os.path.join(fs.VIDEOS, 'Fig5D.mp4')); fD = rD.get_data(rD.count_frames()//3)
    axD = fig.add_axes([.55, y(132), .43, hh(70)]); fs.image_panel(axD, fD); fs.play_badge(fig, axD); fs.fig_letter(fig, .53, y(57), 'D')
    rE = imageio.get_reader(os.path.join(fs.VIDEOS, 'Fig5E.mp4')); fE = rE.get_data(rE.count_frames()//3)
    ew = .9; eh = ew*180*fE.shape[0]/fE.shape[1]; ex = .07
    axE = fig.add_axes([ex, y(144+eh), ew, hh(eh)]); fs.image_panel(axE, fE); fs.play_badge(fig, axE); fs.fig_letter(fig, .005, y(140), 'E')
    fig.text(ex, y(141), 'k = 0 (rotation)', fontsize=7, va='top'); fig.text(ex+ew, y(141), 'k = 1 (wobble)', fontsize=7, ha='right', va='top')
    fig.text(ex-.01, y(144+eh*.25), 'H', fontsize=8, ha='right', va='center'); fig.text(ex-.01, y(144+eh*.75), 'V', fontsize=8, ha='right', va='center')
    axF = fig.add_axes([.1, y(242), .35, hh(26)]); draw_cos(axF, 'U'); fs.letter(axF, 'F', x=-.28)
    axG = fig.add_axes([.6, y(242), .35, hh(26)]); draw_cos(axG, 'A'); fs.letter(axG, 'G', x=-.28)
    fs.save(fig, 'Fig5')
    if '--no-video' in sys.argv: sys.exit()
    # Fig 5C video: the flow over the fitted part of the cycle
    w = imageio.get_writer(os.path.join(fs.VOUT, 'Fig5C.mp4'), fps=10, codec='libx264', quality=8, macro_block_size=1)
    fv = fs.plt.figure(figsize=(7.0, 3.4), dpi=150)                   # even pixel size (1050 x 510) for the encoder
    for fi in range(M.P.FIT.start, M.P.FIT.stop):
        fv.clf(); flow_grid(fv, [.0, .03, .86, .94], fi); draw_key(fv.add_axes([.86, .38, .14, .24]))
        fv.canvas.draw(); w.append_data(np.asarray(fv.canvas.buffer_rgba())[..., :3])
    w.close(); print('saved', os.path.join(fs.VOUT, 'Fig5C.mp4'))
    for v in ('Fig5D.mp4', 'Fig5E.mp4'):
        fs.copy_video(os.path.join(fs.VIDEOS, v), v)
