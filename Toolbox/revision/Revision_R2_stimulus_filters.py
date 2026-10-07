#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stimulus manipulations meant to remove the effect of the cortical anisotropy (new analysis):
  'original'    D4G ring (sigma 1.5 px), as in Figure 5
  'compensated' the video filtered by H(f) = sqrt(S_iso(f) / S_aniso(f)), S(f) = sum_i |G_i(f)|^2 over the 80 filters
                (population sensitivity); designed for the cortex anisotropy beta, clipped to [0.25, 4], DC kept
  'notch'       orientation notch: spatial-frequency directions where the cat widths deviate most from the mean
                (normalised deviation > 0.7: around horizontal edges and the 22.5 / 157.5 deg obliques) removed,
                cosine taper between 0.5 and 0.7; independent of beta
  'dots'        ring made of isotropic band-pass dots (radial D4G, sigma 1.5 px) on 60 material points of the ring,
                moving rigidly with it: every dot drives all orientations equally
"""
import os, sys
import numpy as np
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from spherical_gabor_bank import exact_bank_sym, fold, cat_hwhh, THETAS
from Revision_R2_D4G_ring import d4g_ring_video
from Revision_R2_D4G_fit import d4g, pix
from Revision_R2_two_rings_ME import stimulus_and_fields, TILT, sz_x, sz_y, Omega
from scipy.spatial import cKDTree
F0, SCALE, SIGMA = .12, .5, 1.5
N_DOTS = 60


def bank(beta, f0=F0):
    b = exact_bank_sym(1, f0=f0)
    h = fold(cat_hwhh(beta))*SCALE
    s = np.repeat(2*f0*np.sin(h/2)/np.sqrt(2*np.log(2)), b.S)
    b.C = s[:, None, None]**2*np.eye(3)[None]
    b.w = np.ones_like(b.w)                                          # width anisotropy only (equal numbers)
    return b


def freq_grid(shape):
    T, H, W = shape
    ft = np.fft.fftshift(np.fft.fftfreq(T))[:, None, None]
    fy = -np.fft.fftshift(np.fft.fftfreq(H))[None, :, None]
    fx = np.fft.fftshift(np.fft.fftfreq(W))[None, None, :]
    return ft, fy, fx


from functools import lru_cache


@lru_cache(maxsize=4)
def sensitivity(beta, shape):
    S = np.zeros(shape)
    for G in bank(beta).masks(shape):
        S += G**2
    return S


def apply_gain(video, gain):
    Vf = np.fft.fftshift(np.fft.fftn(video))*gain
    return np.real(np.fft.ifftn(np.fft.ifftshift(Vf)))


def compensated(video, beta):
    if beta == 0:
        return video
    Si, Sa = sensitivity(0.0, tuple(video.shape)), sensitivity(float(beta), tuple(video.shape))
    g = np.ones(video.shape); ok = Sa > 1e-3*Sa.max()
    g[ok] = np.clip(np.sqrt(Si[ok]/Sa[ok]), .25, 4)
    return apply_gain(video, g)


def notch_gain(shape):
    ft, fy, fx = freq_grid(shape)
    ang = np.arctan2(fy, fx)+0*ft                                    # spatial-frequency direction
    h = fold(cat_hwhh(1)); dev = np.abs(h/h.mean()-1); dev = dev/dev.max()
    th = np.concatenate([THETAS, [2*np.pi]]); dv = np.concatenate([dev, dev[:1]])
    d = np.interp(ang % (2*np.pi), th, dv)
    g = np.clip((.7-d)/.2, 0, 1)
    g = .5-.5*np.cos(np.pi*g)
    g[(fx == 0) & (fy == 0)+0*ft.astype(bool)] = 1
    return g


def notched(video):
    return apply_gain(video, notch_gain(video.shape))


def dots_video(orient, sigma=SIGMA, n=N_DOTS):
    pos1, pos2, _, _, _ = stimulus_and_fields(TILT['bottom'], orient)
    idx = np.linspace(0, pos1.shape[1], n, endpoint=False).astype(int)
    video = np.zeros((len(pos1), sz_y, sz_x))
    for t in range(len(pos1)):
        x, y = pos1[t, idx], pos2[t, idx]
        col = (sz_x-1)-(x+1.2)/2.4*(sz_x-1); row = (y+1.2)/2.4*(sz_y-1)
        d = cKDTree(np.stack([row, col], 1)).query(pix)[0].reshape(sz_y, sz_x)
        video[t] = d4g(d, sigma)/3
    return video


def stimulus(kind, orient, beta=0):
    if kind == 'dots':
        return dots_video(orient)
    v = d4g_ring_video(orient, sigma=SIGMA)
    if kind == 'compensated':
        return compensated(v, beta)
    if kind == 'notch':
        return notched(v)
    return v


@lru_cache(maxsize=4)
def sensitivity_rotated(beta, shape):
    """Population sensitivity of the anisotropic cortex rotated by 90 deg in (f_x, f_y): every filter's centre rotated,
    keeping its width (the widths of horizontal-edge filters now sit at vertical edges and so on)."""
    b = bank(beta); mu = b.mu.copy(); b.mu = np.stack([-mu[:, 1], mu[:, 0], mu[:, 2]], 1)
    S = np.zeros(shape)
    for G in b.masks(shape):
        S += G**2
    return S


def rotated_compensation(video, beta):
    """Filter so that the anisotropic cortex sees the video with the population sensitivity of the 90-deg rotated
    cortex: H(f) = sqrt(S_rot(f) / S_aniso(f)), clipped to [0.25, 4]. Applied to the vertically rotating ring, the
    cortex then receives it as it receives the horizontally rotating ring (in population energy)."""
    Sa, Sr = sensitivity(float(beta), tuple(video.shape)), sensitivity_rotated(float(beta), tuple(video.shape))
    g = np.ones(video.shape); ok = Sa > 1e-3*Sa.max()
    g[ok] = np.clip(np.sqrt(Sr[ok]/Sa[ok]), .25, 4)
    return apply_gain(video, g)


def wiggle_video(orient, amp=2.0, ncyc=24, sigma=SIGMA):
    """Option 2: broaden the local orientation content of the contour. The ring's centre line is displaced along its
    image-plane normal by amp * sin(ncyc * theta) px, theta the material angle (the wiggle moves rigidly with the ring),
    so each local edge spans orientations of +- atan(2 pi amp / wavelength) around the ring's own orientation."""
    pos1, pos2, _, _, _ = stimulus_and_fields(TILT['bottom'], orient)
    th0 = np.linspace(0, 2*np.pi, pos1.shape[1]); th = np.linspace(0, 2*np.pi, 4000, endpoint=False)
    video = np.zeros((len(pos1), sz_y, sz_x))
    for t in range(len(pos1)):
        x, y = np.interp(th, th0, pos1[t]), np.interp(th, th0, pos2[t])
        col = (sz_x-1)-(x+1.2)/2.4*(sz_x-1); row = (y+1.2)/2.4*(sz_y-1)
        dr, dc = np.gradient(row), np.gradient(col); n = np.hypot(dr, dc)+1e-12
        off = amp*np.sin(ncyc*th)
        line = np.stack([row+off*dc/n, col-off*dr/n], 1)
        video[t] = d4g(cKDTree(line).query(pix)[0].reshape(sz_y, sz_x), sigma)/3
    return video


def dashed_video(orient, depth=1.0, ncyc=24, sigma=SIGMA):
    """Smooth oval with luminance modulated along the contour: D4G profile x (1 - depth/2 + depth/2 sin(ncyc theta)),
    theta the material angle of the nearest centre-line point (the pattern rotates rigidly with the ring). depth 1:
    fully dashed (ncyc dashes); smaller: beaded."""
    pos1, pos2, _, _, _ = stimulus_and_fields(TILT['bottom'], orient)
    th0 = np.linspace(0, 2*np.pi, pos1.shape[1]); th = np.linspace(0, 2*np.pi, 4000, endpoint=False)
    mod = 1-depth/2+depth/2*np.sin(ncyc*th)
    video = np.zeros((len(pos1), sz_y, sz_x))
    for t in range(len(pos1)):
        x, y = np.interp(th, th0, pos1[t]), np.interp(th, th0, pos2[t])
        line = np.stack([(y+1.2)/2.4*(sz_y-1), (sz_x-1)-(x+1.2)/2.4*(sz_x-1)], 1)
        d, i = cKDTree(line).query(pix)
        video[t] = (d4g(d, sigma)*mod[i]).reshape(sz_y, sz_x)/3
    return video


def texture_video(orient, width=8.0, density=.3, dot_sigma=1.0, seed=0, streak=0.0, lifetime=0, fade=False, slide=0.0):
    """Smooth oval rim made of random-dot texture: dots with fixed material coordinates (angle theta along the ring,
    offset across it, uniform within a band of `width` px), moving rigidly with the ring; each dot a Gaussian blob of
    SD dot_sigma px, random polarity; density in dots per px^2 of band.
    streak > 0: control that keeps the aperture problem: every dot stretched along the ring into an arc parallel to
    the rim (Gaussian taper, SD `streak` px of arc length), so the texture's orientation energy is that of the contour
    and its motion along the rim is (nearly) invisible; elements overlap into oriented noise along the rim.
    lifetime > 0 (frames): limited-lifetime dots, so no dot can be tracked: each dot moves rigidly with the ring for
    `lifetime` frames and then reappears at new random material coordinates; lifetimes are staggered.
    fade: each element's contrast rises and falls over its lifetime (sin^2 of its age), so the texture changes
    continuously, without on/off transients.
    slide: the texture moves with motion mix k = slide instead of rigidly: the ring's outline is the same (an in-plane
    rotation of the ring, X_k = Ry(Omega) R_n(-k Omega) p), but each element slides along it, sitting at ring parameter
    theta + k*Omega (Revision_R2_two_rings.ring_motion); slide = 1 is the wobble template."""
    from scipy.ndimage import gaussian_filter
    pos1, pos2, _, _, _ = stimulus_and_fields(TILT['bottom'], orient)
    th0 = np.linspace(0, 2*np.pi, pos1.shape[1])
    circ = np.sum(np.hypot(np.diff(pos1[0]), np.diff(pos2[0])))/2.4*(sz_x-1)
    n = int(density*width*circ)                                     # streaks: same count, overlapping (oriented noise)
    rng = np.random.default_rng(seed)
    tk, ok, sk = rng.uniform(0, 2*np.pi, n), rng.uniform(-width/2, width/2, n), rng.choice([-1., 1.], n)
    if lifetime:                                                     # coordinates for every generation of every dot
        G = len(pos1)//lifetime+2; ph = rng.integers(0, lifetime, n)
        TK, OK, SK = rng.uniform(0, 2*np.pi, (G, n)), rng.uniform(-width/2, width/2, (G, n)), rng.choice([-1., 1.], (G, n))
    def elongate(tk, ok, sk):                                       # streak: arc samples along the ring, Gaussian weights
        if not streak:
            return tk, ok, sk
        ds = np.arange(-3*streak, 3*streak+.01, .5); wt = np.exp(-ds**2/(2*streak**2))
        return ((tk[:, None]+ds[None]/circ*2*np.pi).ravel() % (2*np.pi), np.repeat(ok, len(ds)),
                (sk[:, None]*wt[None]).ravel()*.5/(np.sqrt(2*np.pi)*streak)*2*np.sqrt(2*np.pi)*dot_sigma)   # peak ~ a dot
    tk0, ok0, sk0 = elongate(tk, ok, sk)
    video = np.zeros((len(pos1), sz_y, sz_x))
    for t in range(len(pos1)):
        tk, ok, sk = tk0, ok0, sk0
        if lifetime:                                                 # each element from its current generation
            g = (t+ph)//lifetime; a_ = np.arange(n); sk = SK[g, a_]
            if fade:
                sk = sk*np.sin(np.pi*(((t+ph) % lifetime)+.5)/lifetime)**2*2   # mean contrast as without fading
            tk, ok, sk = elongate(TK[g, a_], OK[g, a_], sk)
        if slide:
            tk = (tk+slide*Omega[t]) % (2*np.pi)
        x, y = np.interp(tk, th0, pos1[t]), np.interp(tk, th0, pos2[t])
        dx, dy = np.interp((tk+1e-3) % (2*np.pi), th0, pos1[t])-x, np.interp((tk+1e-3) % (2*np.pi), th0, pos2[t])-y
        row, col = (y+1.2)/2.4*(sz_y-1), (sz_x-1)-(x+1.2)/2.4*(sz_x-1)
        dr, dc = dy, -dx; nn = np.hypot(dr, dc)+1e-12                   # tangent in (row, col)
        row, col = row+ok*dc/nn, col-ok*dr/nn                            # offset along the normal
        img = np.zeros((sz_y, sz_x)); r0, c0 = np.floor(row).astype(int), np.floor(col).astype(int); fr, fc = row-r0, col-c0
        for drr, dcc, w in ((0, 0, (1-fr)*(1-fc)), (1, 0, fr*(1-fc)), (0, 1, (1-fr)*fc), (1, 1, fr*fc)):
            rr, cc = np.clip(r0+drr, 0, sz_y-1), np.clip(c0+dcc, 0, sz_x-1)
            np.add.at(img, (rr, cc), w*sk)
        video[t] = gaussian_filter(img, dot_sigma)*2*np.pi*dot_sigma**2
    return video
