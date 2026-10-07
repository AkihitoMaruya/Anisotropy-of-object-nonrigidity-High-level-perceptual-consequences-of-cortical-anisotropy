#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary: the motion-energy filters (three radial bands, f0 = 0.065, 0.12, 0.22) and the ring-video spectrum
projected onto the (f_x, f_y) plane.
Only the f_t < 0 half is used (the spectrum of a real video is point-symmetric), so the filters there are those that
prefer motion along +theta, and a lobe at angle theta in (f_x, f_y) is the unit preferring motion direction theta.
  filters: for each direction theta, the sum over f_t < 0 of its moving channels (phi = -30, -60 deg) of the Figure 5
           model (isotropic cortex; anisotropic cortex: full-width cat widths x 2.5); filled at half height, colour =
           preferred motion direction as in Figure 5 (left red, up purple, right cyan, down green).
  ring:    |FFT|^2 of the D4G ring video summed over f_t < 0 (whole cycle), within 0.03 < |f| < 0.19 (the filters' band),
           log scale, darker = more energy.
Rows: isotropic / anisotropic cortex; columns: filters only / + horizontal rotation / + vertical rotation.
New analysis.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from spherical_gabor_bank import exact_bank_sym, fold, cat_hwhh, LAYOUT_C, THETAS
from Revision_R2_D4G_ring import d4g_ring_video
out = current_folder+'/Toolbox/Data/work/'
BETA, SCALE, LIM = 2.5, .5, .22
F0S = [.12]                                                       # the single radial band of the Figure 5 model
plt.rcParams.update({'font.family': 'Arial', 'font.size': 24, 'axes.titlesize': 26, 'axes.labelsize': 26})


def bank(kind, f0):
    b = exact_bank_sym(1 if kind == 'A' else 0, f0=f0)
    h = fold(cat_hwhh(BETA if kind == 'A' else 0))*SCALE
    s = np.repeat(2*f0*np.sin(h/2)/np.sqrt(2*np.log(2)), b.S)
    b.C = s[:, None, None]**2*np.eye(3)[None]
    return b


def dir_rgb(a):
    hue = (np.arctan2(-np.sin(a), -np.cos(a))/(2*np.pi)) % 1          # as in Figure 5
    return hsv_to_rgb([hue, 1, .9])


g = np.linspace(-LIM, LIM, 301); FX, FY = np.meshgrid(g, g, indexing='xy')


def masks2d(kind, f0):
    """per direction: sum over f_t < 0 of its moving-channel Gaussians (analytic f_t integral of a spherical Gaussian:
    a 2D Gaussian of the same SD around mu_s, weighted by the fraction of its mass at f_t < 0)."""
    b = bank(kind, f0); M = np.zeros((16,)+FX.shape)
    from scipy.special import erfc
    for k in range(16):
        for j in np.where(LAYOUT_C < 0)[0]:
            i = k*b.S+j; mu = b.mu[i]; s = np.sqrt(b.C[i, 0, 0])
            w = .5*erfc(mu[2]/(s*np.sqrt(2)))                         # mass at f_t < 0
            M[k] += w*s*np.exp(-((FX-mu[0])**2+(FY-mu[1])**2)/(2*s**2))
    return M


def spectrum2d(o):
    V = d4g_ring_video(o, sigma=1.5); T, H, W = V.shape
    S = np.abs(np.fft.fftshift(np.fft.fftn(V)))**2
    ft = np.fft.fftshift(np.fft.fftfreq(T)); fy = -np.fft.fftshift(np.fft.fftfreq(H)); fx = np.fft.fftshift(np.fft.fftfreq(W))
    P = S[ft < 0].sum(0)                                              # (fy, fx), fy decreasing down the rows
    R = np.hypot(*np.meshgrid(fy, fx, indexing='ij'))
    P = P*((R > .03) & (R < .19))                                     # the band covered by the filters
    L = np.log10(P/P.max()+1e-12)
    return L, (fx[0], fx[-1], fy[-1], fy[0])


SPEC = {o: spectrum2d(o) for o in ('H', 'V')}
MASK = {(kind, f0): masks2d(kind, f0) for kind in ('U', 'A') for f0 in F0S}
fig, ax = plt.subplots(2, 3, figsize=(24, 16.5))
for r, kind in enumerate(('U', 'A')):
    for c, stim in enumerate((None, 'H', 'V')):
        a = ax[r, c]
        if stim:
            L, ext = SPEC[stim]
            a.imshow(L, extent=ext, cmap='Greys', vmin=-2, vmax=0, origin='upper', interpolation='bilinear')
        for f0 in F0S:
            for k in range(16):
                m = MASK[(kind, f0)][k]; col = dir_rgb(THETAS[k])
                a.contourf(FX, FY, m, levels=[.5*m.max(), m.max()*1.01], colors=[col], alpha=.45 if stim else .75)
                a.contour(FX, FY, m, levels=[.5*m.max()], colors=[col], linewidths=2.5)
        a.set(xlim=[-LIM, LIM], ylim=[-LIM, LIM], aspect='equal', xticks=[-.2, 0, .2], yticks=[-.2, 0, .2])
        a.set_xlabel(r'$f_x$ (cycles/px)'); a.set_ylabel(r'$f_y$ (cycles/px)')
        a.set_title(('Isotropic' if kind == 'U' else 'Anisotropic')+' cortex'+('' if stim is None else
                    f'\n+ ring, {"horizontal" if stim == "H" else "vertical"} rotation'))
# direction key
ka = fig.add_axes([.905, .43, .09, .14]); ang = np.radians(np.arange(0, 360, 45)); v = np.stack([np.cos(ang), np.sin(ang)], 1)
ka.quiver(np.zeros(8), np.zeros(8), v[:, 0], v[:, 1], color=[dir_rgb(x) for x in ang], angles='xy', scale_units='xy', scale=1,
          width=.05, headwidth=3, headlength=3.5)
for (x, y), t in zip(v[::2]*1.55, ('right', 'up', 'left', 'down')):
    ka.text(x, y, t, ha='center', va='center', fontsize=18)
ka.set(xlim=[-2.1, 2.1], ylim=[-2.1, 2.1], aspect='equal'); ka.axis('off')
fig.text(.46, .005, 'Coloured: half-height region of each direction-selective unit (moving channels, summed over $f_t<0$); colour = preferred motion direction.\n'
         'Grey: ring-video power spectrum summed over $f_t<0$ (log; darker = more energy), within the filters\' frequency band.', ha='center', fontsize=19)
plt.tight_layout(rect=[0, .04, .9, 1]); fig.savefig(out+'R2_86_figS_filters_fxfy.png', dpi=75)
print('ok')
