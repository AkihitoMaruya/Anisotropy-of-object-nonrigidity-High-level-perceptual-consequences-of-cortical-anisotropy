#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5 A, B, C, F, G with the revised model (the original Figure5*.py are untouched):
Gaussian Heeger motion energy, spherical filters with equal peaks, cat widths read as full width at half height
(Li et al. 2003), folded, width anisotropy beta = 2.5, folded cat cell numbers (x ALPHA = 1)
(isotropic cortex: equal widths and numbers), three radial bands (f0 = 0.065, 0.12, 0.22 cycles/px; costs summed; Revision_R2_three_band.py), plain
Heeger decoding (argmin), no smoothness constraint (optional: HS_ALPHA, Horn-Schunck as in
Compute_optic_flow_from_3D_gabor.py, Revision_R2_fig5_hs_smooth.py), D4G ring.
  A, B  direction tuning for a grating drifting at 0.58 px/frame, in screen coordinates: each direction's energy (sum of
        its two moving temporal channels that prefer motion along +theta), peak 1, times the cell number (B: mean number)
  C     optic flow at 45 deg phase (horizontal / vertical rotation x isotropic / anisotropic)
  F, G  cosine similarity between flow and template directions vs k (mean +- std/2), best k
  D, E  snapshots of the original Videos/Fig5D.mp4 and Fig5E.mp4 (unchanged), for the combined figure only
  supplementary figure: best-fitting k vs anisotropy strength: tuning width (equal numbers) and cell numbers (equal widths), 0-5 x cat
        (flows from Revision_R2_width_vs_number_sweep.py)
Outputs (Images/Revision_R2): R2_86_fig5{A,B,C,F,G}.png, R2_86_figS_anisotropy_strength.png, R2_86_fig5C.mp4 (flow over the cycle), R2_86_fig5_combined.png.
Colour = motion direction, the same in A, B, C and as in the original 5E: leftward red, upward purple, rightward cyan,
downward green.
New analysis; reads the existing code and data only.
"""
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
os.environ.setdefault('K_FIT_TRIM', '10')                        # frames 5-122 are decoded; dropping 10 more at each end
#   leaves video frames 15-112 (first and last 15 of the 128-frame cycle removed: occlusion near edge-on)
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from spherical_gabor_bank import exact_bank_sym, THETAS, LAYOUT_C, fold, cat_hwhh
from anisotropy_heeger import cat_counts
import Revision_R2_heeger_temperature_sweep as TS
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import Omega, sz_x, sz_y
from Revision_R2_D4G_ring import d4g_ring_video
D, P = TS.D, TS.P
BETA, f0, SCALE = 2.5, .12, .5
ALPHA = 1                                                        # cell-number anisotropy (x cat; 1 = cat numbers)
HS_ALPHA = 1.0                                                   # smoothness constraint weight (x median speed); 0 = none
THREE_BANDS = False                                              # three radial bands f0 = 0.065, 0.12, 0.22 (Revision_R2_three_band.py)
NUM = 1+ALPHA*(fold(cat_counts(THETAS))-1)                       # folded cat numbers (mean 1), anisotropy x ALPHA
out = P.A.out


def bank(kind):
    b = exact_bank_sym(1 if kind == 'A' else 0)                      # 'A': anisotropic numbers; 'U': equal numbers
    h = fold(cat_hwhh(BETA if kind == 'A' else 0))*SCALE
    s = np.repeat(2*f0*np.sin(h/2)/np.sqrt(2*np.log(2)), b.S)
    b.C = s[:, None, None]**2*np.eye(3)[None]
    if kind == 'A':
        b.w = np.repeat(NUM, b.S)
    return b


def dir_rgb(V):
    """Direction colour as in the original Figure 5E (Vis_vec_field: hue = atan2(-v, -u) / 2 pi, screen y up):
    leftward red, downward yellow-green, rightward cyan, upward purple."""
    hue = (np.arctan2(-V[..., 1], -V[..., 0])/(2*np.pi)) % 1
    return hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.full_like(hue, .9)], -1))


def direction_key(ax):
    ang = np.radians(np.arange(0, 360, 45)); v = np.stack([np.cos(ang), np.sin(ang)], 1)
    ax.quiver(np.zeros(8), np.zeros(8), v[:, 0], v[:, 1], color=dir_rgb(v), angles='xy', scale_units='xy', scale=1,
              width=.035, headwidth=3.5, headlength=4)
    for (x, y), t in zip(v[::2]*1.45, ('right', 'up', 'left', 'down')):
        ax.text(x, y, t, ha='center', va='center', fontsize=12)
    ax.set(xlim=[-1.9, 1.9], ylim=[-1.9, 1.9], aspect='equal'); ax.axis('off')


# flows
F = {}
for o in ('H', 'V'):
    cache = P.A.data+f'heeger_fig5_revised_{o}.npz'
    Z = dict(np.load(cache)) if os.path.exists(cache) else {}
    video = None
    for kind in ('U', 'A'):
        key = kind if (kind == 'U' or ALPHA == 1) else f'A_n{ALPHA}'      # 'A': cat numbers
        if key not in Z:
            t0 = time.time()
            if video is None:
                video = d4g_ring_video(o, sigma=1.5); Vf = np.fft.fftshift(np.fft.fftn(video))
                L, _, _ = ring_points(o); L = L[FRAMES]
                pts = (np.repeat(FRAMES, NP), np.clip(np.round(L[..., 0]), 0, sz_y-1).astype(int).ravel(),
                       np.clip(np.round(L[..., 1]), 0, sz_x-1).astype(int).ravel())
            b = bank(kind)
            Z[key] = b.flow(b.energies(video, pts, Vf)).reshape(len(FRAMES), NP, 2); np.savez(cache, **Z)
            print(f'  {o} {key}: {(time.time()-t0)/60:.1f} min', flush=True)
        F[(o, kind)] = Z[key]
    if THREE_BANDS:                                                  # three radial bands (Revision_R2_three_band.py)
        tb = np.load(P.A.data+f'heeger_three_band_{o}.npz')
        for kind in ('U', 'A'):
            F[(o, kind)] = tb[kind+('_hs' if HS_ALPHA else '')]
    elif HS_ALPHA:                                                   # the paper's smoothness constraint (Horn-Schunck)
        hsz = np.load(P.A.data+f'heeger_fig5_hs_{o}.npz')             # cached by Revision_R2_fig5_hs_smooth.py
        for kind in ('U', 'A'):
            F[(o, kind)] = hsz[f'{kind}_a{HS_ALPHA}']

import imageio.v2 as imageio
VID = current_folder+'/Toolbox/Data/videos/'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 26, 'axes.linewidth': 2.5, 'xtick.major.width': 2.5,
                     'ytick.major.width': 2.5, 'xtick.major.size': 8, 'ytick.major.size': 8, 'axes.titlesize': 30,
                     'axes.labelsize': 28, 'legend.fontsize': 21, 'xtick.labelsize': 24, 'ytick.labelsize': 24})
LW = 4.5
alpha = np.radians(np.arange(-180, 180, 1.0))
PHI_S = np.radians(30)
fg = f0*np.stack([np.cos(PHI_S)*np.cos(alpha), np.cos(PHI_S)*np.sin(alpha), -np.full_like(alpha, np.sin(PHI_S))], 1)
VIDS = {o: d4g_ring_video(o, sigma=1.5) for o in ('H', 'V')}
LP = {o: ring_points(o)[0][FRAMES] for o in ('H', 'V')}
FI45 = int(np.argmin(np.abs(np.degrees(Omega[FRAMES])-45)))


def draw_tuning(ax, kind):
    """Direction tuning for a grating drifting at 0.58 px/frame (screen coordinates, f_t = -f_s . v); energy of the
    two moving channels that prefer motion along +theta; peak 1 x cell number."""
    b = bank(kind)
    G = np.zeros((len(b.mu), len(alpha)))
    for i, (mu, C) in enumerate(zip(b.mu, b.C)):
        G[i] = np.exp(-np.sum((fg-mu)**2, 1)/C[0, 0])+np.exp(-np.sum((-fg-mu)**2, 1)/C[0, 0])
    R = G.reshape(16, b.S, -1)[:, LAYOUT_C < 0].sum(1)
    R = R/R.max(1, keepdims=True)
    n = NUM if kind == 'A' else np.full(16, NUM.mean())
    for k in range(16):
        pk = alpha[np.argmax(R[k])]
        ax.plot(np.degrees(alpha), n[k]*R[k], color=dir_rgb(np.array([np.cos(pk), np.sin(pk)])), lw=LW)
    ax.set(xticks=np.arange(-180, 181, 90), xlim=[-180, 180], ylim=[0, 1.4], xlabel='Motion direction (deg)',
           ylabel=r'$n_i m_i$' if kind == 'A' else r'$\bar{n} m_i$', title='ME anisotropy' if kind == 'A' else 'ME isotropy')


def draw_flow(ax, o, kind, fi, title=True, fs=20):
    V = F[(o, kind)][fi]; Pp = LP[o][fi]; u = V/np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9); s = slice(None, None, 6)
    ax.imshow(VIDS[o][FRAMES[fi]], cmap='gray', vmin=-1.5, vmax=1.5, alpha=.5)
    ax.quiver(Pp[s, 1], Pp[s, 0], u[s, 0], -u[s, 1], color=dir_rgb(V[s]), angles='xy', scale_units='xy', scale=.025,
              width=.010, headwidth=3.2, headlength=3.8)          # every 6th contour point, arrows 40 px long
    if title:
        ax.set_title(f'{"Horizontal" if o == "H" else "Vertical"} rotation\n{"isotropic" if kind == "U" else "anisotropic"} cortex', fontsize=fs)
    ax.axis('off')


def draw_key(ax, fs=18):
    ang = np.radians(np.arange(0, 360, 45)); v = np.stack([np.cos(ang), np.sin(ang)], 1)
    ax.quiver(np.zeros(8), np.zeros(8), v[:, 0], v[:, 1], color=dir_rgb(v), angles='xy', scale_units='xy', scale=1,
              width=.05, headwidth=3, headlength=3.5)
    for (x, y), t in zip(v[::2]*1.55, ('right', 'up', 'left', 'down')):
        ax.text(x, y, t, ha='center', va='center', fontsize=fs)
    ax.set(xlim=[-2.1, 2.1], ylim=[-2.1, 2.1], aspect='equal'); ax.axis('off')


def draw_flow_grid(fig, rect, fi, titles=True, fs=20):
    """2 x 2 flows (columns: horizontal, vertical rotation; rows: isotropic, anisotropic) inside rect [x, y, w, h]."""
    x0, y0, w, h = rect
    for c, o in enumerate(('H', 'V')):
        for r, kind in enumerate(('U', 'A')):
            draw_flow(fig.add_axes([x0+c*w/2, y0+(1-r)*h/2, w/2*.98, h/2*.86]), o, kind, fi, titles, fs)


COS = {}
for kind in ('U', 'A'):
    for o in ('H', 'V'):
        U_, V_ = D.flow_phys(F[(o, kind)]); u, v = D.unit(U_, V_); ok = np.isfinite(u) & (np.hypot(U_, V_) > 1e-9)
        cs = [(u*T[0]+v*T[1])[P.FIT][ok[P.FIT]] for T in D.TU[o]]
        COS[(kind, o)] = (np.array([c.mean() for c in cs]), np.array([c.std() for c in cs]))
K = D.KG; SEL = K <= 1.0


def draw_cos(ax, kind, legend=True):
    for o, col, fc, nm, lw in (('V', 'r', 'deeppink', 'V', LW+3.5), ('H', 'b', 'lightblue', 'H', LW+1)):   # H on top (they coincide when isotropic)
        m, sd_ = COS[(kind, o)]; m, sd_, k = m[SEL], sd_[SEL], K[SEL]; kb = k[np.argmax(m)]
        ax.plot(k, m, col+'-', lw=lw, label=f'{nm} mean')
        ax.fill_between(k, m-sd_/2, m+sd_/2, alpha=.2, color=fc)
        ax.axvline(kb, ls='--', color=col, lw=3, label=f'{nm} max, k = {kb:.2f}')
    ax.set(xticks=np.arange(0, 1.1, .2), yticks=[0, .5, 1], ylim=[0, 1.05], xlabel=r'$k$', ylabel='Cosine similarity\nto template',
           title='Isotropic cortex' if kind == 'U' else 'Anisotropic cortex')
    ar = dict(arrowstyle='-|>', lw=3, color='k', mutation_scale=25)                # rotation <- k -> wobble
    ax.annotate('', xy=(0.0, -.26), xytext=(.36, -.26), xycoords='axes fraction', arrowprops=ar, annotation_clip=False)
    ax.annotate('', xy=(1.0, -.26), xytext=(.64, -.26), xycoords='axes fraction', arrowprops=ar, annotation_clip=False)
    ax.text(.18, -.24, 'Rotation', transform=ax.transAxes, ha='center', va='bottom', fontsize=24)
    ax.text(.82, -.24, 'Wobble', transform=ax.transAxes, ha='center', va='bottom', fontsize=24)
    if legend:
        h, l = ax.get_legend_handles_labels(); order = [2, 3, 0, 1]
        ax.legend([h[i] for i in order], [l[i] for i in order], loc='lower center')


def snapshot(name, frac=1/3):
    r = imageio.get_reader(VID+name); n = r.count_frames(); return r.get_data(int(n*frac))


PLAY = imageio.imread(VID+'playmark.png')


def play_badge(fig, ax, size=.035):
    p = ax.get_position(); a = fig.add_axes([p.x1-size*1.1, p.y0+size*.1, size, size*fig.get_figwidth()/fig.get_figheight()])
    a.imshow(PLAY); a.axis('off')


# separate panels
for tag, kind in (('A', 'A'), ('B', 'U')):
    fig = plt.figure(figsize=(12, 7)); ax = fig.add_axes([.12, .16, .66, .74]); draw_tuning(ax, kind)
    draw_key(fig.add_axes([.79, .3, .2, .4])); fig.savefig(out+f'R2_86_fig5{tag}.png', dpi=100); plt.close(fig)
for tag, kind in (('F', 'U'), ('G', 'A')):
    fig = plt.figure(figsize=(10, 10)); draw_cos(fig.add_axes([.2, .27, .75, .64]), kind); fig.savefig(out+f'R2_86_fig5{tag}.png', dpi=90); plt.close(fig)
fig = plt.figure(figsize=(16, 14)); draw_flow_grid(fig, [.01, .01, .84, .95], FI45); draw_key(fig.add_axes([.85, .4, .14, .2]))
fig.savefig(out+'R2_86_fig5C.png', dpi=80); plt.close(fig)
# Figure 5C video
w = imageio.get_writer(out+'R2_86_fig5C.mp4', fps=10, codec='libx264', quality=8, macro_block_size=1)
fig = plt.figure(figsize=(12, 10.5), dpi=80)
for fi in range(P.FIT.start, P.FIT.stop):                               # same frames as the fits
    fig.clf(); draw_flow_grid(fig, [.01, .01, .84, .95], fi, fs=16); draw_key(fig.add_axes([.85, .4, .14, .2]), fs=13)
    fig.canvas.draw(); w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
w.close(); plt.close(fig)
# supplementary figure: anisotropy strength vs best k (Revision_R2_width_vs_number_sweep.py, cached flows)
LEVELS = np.round(np.arange(0, 5.01, .5), 1)
KCACHE = P.A.data+'heeger_three_band_sweep_k.npz'                 # three bands (Revision_R2_three_band_sweep.py)
if THREE_BANDS and os.path.exists(KCACHE):
    kz = np.load(KCACHE)
    KS = {(o, kd, lv): kz[f'{o}_{kd}'][i] for o in ('H', 'V') for kd in ('w', 'n') for i, lv in enumerate(LEVELS)}
else:                                                                # one band, unsmoothed (Revision_R2_width_vs_number_sweep.py)
    SW = {o: np.load(P.A.data+f'heeger_width_vs_number_{o}.npz') for o in ('H', 'V')}
    KS = {(o, kd, lv): D.fit3(*D.flow_phys(SW[o][f'{kd}{lv}']), o)[0] for o in ('H', 'V') for kd in ('w', 'n') for lv in LEVELS}


def draw_strength(ax, kd, ylabel=True):
    ax.plot(LEVELS, [KS[('H', kd, l)] for l in LEVELS], 'b-o', lw=LW+.5, ms=11, label='Horizontal rotation')
    ax.plot(LEVELS, [KS[('V', kd, l)] for l in LEVELS], 'r-o', lw=LW+.5, ms=11, label='Vertical rotation')
    ax.axvline(1, color='gray', ls=':', lw=3); ax.text(1.08, 1.0, 'cat', color='gray', fontsize=20)
    ax.set(xticks=range(6), ylim=[0, 1.08], xlabel=('Tuning-width' if kd == 'w' else 'Cell-number')+' anisotropy (× cat)',
           title='Width anisotropy, equal numbers' if kd == 'w' else 'Number anisotropy, equal widths')
    if ylabel:
        ax.set_ylabel('Best-fitting k')
    ax.legend(loc='lower left')


fig, ax = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
draw_strength(ax[0], 'w'); draw_strength(ax[1], 'n', False); plt.tight_layout(); fig.savefig(out+'R2_86_figS_anisotropy_strength.png', dpi=90); plt.close(fig)

# combined figure A-G (32 in tall)
FH = 32.
Y = lambda y: (y*32+(FH-32))/FH
Hh = lambda h: h*32/FH
fig = plt.figure(figsize=(22, FH))
A_ = lambda x, y, w, h: fig.add_axes([x, Y(y), w, Hh(h)])
L = lambda x, y, t: fig.text(x, Y(y), t, fontsize=34, fontweight='bold', va='top')
draw_tuning(A_(.07, .855, .36, .125), 'A'); L(.01, .99, 'A')
draw_tuning(A_(.55, .855, .36, .125), 'U'); L(.49, .99, 'B')
draw_key(A_(.915, .87, .085, .085), fs=18)
draw_flow_grid(fig, [.02, Y(.48), .47, Hh(.33)], FI45, fs=22); L(.01, .82, 'C')
axC = A_(.02, .48, .47, .33); axC.axis('off'); play_badge(fig, axC)
axD = A_(.53, .50, .44, .30); axD.imshow(snapshot('Fig5D.mp4')); axD.axis('off'); L(.51, .82, 'D'); play_badge(fig, axD)
EW = .70; EH = EW*22/2.5/32                                          # Fig5E frames are 1000 x 400 px
axE = A_((1-EW)/2, .255, EW, EH); axE.imshow(snapshot('Fig5E.mp4')); axE.axis('off'); L(.01, .255+EH+.02, 'E'); play_badge(fig, axE)
fig.text((1-EW)/2+.04, Y(.255+EH+.004), 'k = 0 (rotation)', fontsize=26); fig.text((1+EW)/2-.04, Y(.255+EH+.004), 'k = 1 (wobble)', fontsize=26, ha='right')
fig.text((1-EW)/2-.01, Y(.255+EH*.75), 'H', fontsize=28, ha='right', va='center'); fig.text((1-EW)/2-.01, Y(.255+EH*.25), 'V', fontsize=28, ha='right', va='center')
draw_cos(A_(.10, .075, .36, .145), 'U'); L(.01, .235, 'F')
draw_cos(A_(.60, .075, .36, .145), 'A'); L(.51, .235, 'G')
fig.savefig(out+'R2_86_fig5_combined.png', dpi=80); plt.close(fig)
print('best k: isotropic H / V', K[SEL][np.argmax(COS[("U", "H")][0][SEL])], K[SEL][np.argmax(COS[("U", "V")][0][SEL])],
      '| anisotropic H / V', K[SEL][np.argmax(COS[("A", "H")][0][SEL])], K[SEL][np.argmax(COS[("A", "V")][0][SEL])])
