#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Videos of the experimental two-ring stimulus (two rings tilted +-30 deg and joined, as in Make_rotating_two_rings.py;
D4G contours, sigma 1.5 px) rotating
horizontally and vertically, side by side, after the stimulus manipulations of Revision_R2_stimulus_filters.py, for
viewing whether the wobble percept changes: original; compensating filter designed for beta = 0.5 ... 5 (x cat);
orientation notch; dotted rings. One mp4 per setting (Images/Revision_R2/stimfilter_videos/) and an HTML page with a
stimulus menu and a compensation slider: Images/Revision_R2/R2_86_figS_two_ring_filters.html. New analysis.
"""
import os, sys, json
import numpy as np
import imageio.v2 as imageio
from scipy.spatial import cKDTree
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_stimulus_filters as SF
from Revision_R2_two_rings_ME import sz_x, sz_y
from Revision_R2_D4G_fit import d4g, pix
OUT = current_folder+'/Toolbox/Data/work/'
VD = OUT+'stimfilter_videos/'; os.makedirs(VD, exist_ok=True)
LEVELS = np.round(np.arange(.5, 5.01, .5), 1)


N_FR, PHI, DC = 256, np.radians(30), 100.                         # 360 deg in 256 frames (1.4 deg/frame, as the model ring)


def ring_xy(t, orient, n):
    """The experimental stimulus (Make_rotating_two_rings.py, 'Rot'): two rings tilted +-30 deg, joined where they touch,
    rotating about the vertical axis with perspective (distance 100); vertical rotation = image rotated by 90 deg.
    Returns pixel (row, col) of n material points per ring."""
    th = np.linspace(0, 2*np.pi, n, endpoint=False); Om = -2*np.pi*t/N_FR
    out = []
    for ph, dy in ((PHI, -np.sin(PHI)), (-PHI, np.sin(PHI))):
        X = np.cos(Om)*np.cos(th)-np.sin(Om)*np.sin(th)*np.cos(ph)
        Y = np.sin(th)*np.sin(ph)+dy
        Z = -np.sin(Om)*np.cos(th)-np.cos(Om)*np.sin(th)*np.cos(ph)
        u, v = X*DC/(DC-Z), Y*DC/(DC-Z)
        if orient == 'V':
            u, v = -v, u
        out.append(np.stack([(1-(v+1.1)/2.2)*(sz_y-1), (u+1.1)/2.2*(sz_x-1)], 1))
    return out


def two_ring_video(orient, dots=False, sigma=SF.SIGMA, wiggle=0.0, ncyc=24):
    """wiggle: contour displaced along its image normal by wiggle * sin(ncyc * theta) px (theta material angle, so the
    wiggle moves rigidly with the rings)."""
    v = np.zeros((N_FR, sz_y, sz_x))
    th = np.linspace(0, 2*np.pi, 4000, endpoint=False)
    for t in range(N_FR):
        rings = ring_xy(t, orient, SF.N_DOTS if dots else 4000)
        if wiggle:
            for i, r in enumerate(rings):
                d = np.gradient(r, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True)+1e-12
                rings[i] = r+wiggle*np.sin(ncyc*th)[:, None]*np.stack([d[:, 1], -d[:, 0]], 1)
        line = np.concatenate(rings)
        v[t] = d4g(cKDTree(line).query(pix)[0].reshape(sz_y, sz_x), sigma)/3
    return v


def two_ring_texture_video(orient, width=8.0, density=.3, dot_sigma=1.0, seed=0, streak=0.0, lifetime=0, slide=0.0):
    """Rims as random-dot texture bands (Revision_R2_stimulus_filters.texture_video, here for the joined two rings):
    dots with fixed material coordinates (angle along the ring, offset across an 8 px band), random polarity.
    slide: the dots move with motion mix k = slide (1 = wobble): same outline, each dot sliding along its ring by the
    in-plane rotation R_n(-k Omega) about the ring's upward normal, i.e. to ring parameter theta - k*Om here."""
    from scipy.ndimage import gaussian_filter
    n_line = 4000; rng = np.random.default_rng(seed)
    r0 = ring_xy(0, orient, n_line); circ = sum(np.sum(np.linalg.norm(np.diff(r, axis=0), axis=1)) for r in r0)
    n = int(density*width*circ/2)
    G = N_FR//lifetime+2 if lifetime else 1                          # generations (limited lifetime: staggered rebirths)
    gens = [[(rng.integers(0, n_line, n), rng.uniform(-width/2, width/2, n), rng.choice([-1., 1.], n)) for _ in range(2)]
            for _ in range(G)]
    phase = rng.integers(0, max(lifetime, 1), (2, n))
    dots = gens[0]
    if streak:                                                       # elongate each dot along its ring (arc, Gaussian taper)
        step = circ/2/n_line; ds = np.arange(-3*streak, 3*streak+.01, .5); wt = np.exp(-ds**2/(2*streak**2))
        di = np.round(ds/step).astype(int); g = .5/(np.sqrt(2*np.pi)*streak)*2*np.sqrt(2*np.pi)*dot_sigma
        el = lambda d: [(((ik[:, None]+di[None]) % n_line).ravel(), np.repeat(ok, len(ds)), (sk[:, None]*wt[None]).ravel()*g)
                        for ik, ok, sk in d]
    else:
        di = np.zeros(1, int); el = lambda d: d
    v = np.zeros((N_FR, sz_y, sz_x))
    for t in range(N_FR):
        img = np.zeros((sz_y, sz_x))
        if lifetime:                                                 # each dot from its current generation
            gi = (t+phase)//lifetime; a_ = np.arange(n)
            cur = [tuple(np.stack([gens[gg][ri][q][a] for gg, a in zip(gi[ri], a_)]) for q in range(3)) for ri in range(2)]
        else:
            cur = gens[0]
        sh = int(round(slide*t*n_line/N_FR))                         # -k*Om in samples (Om = -2 pi t / N_FR)
        for r, (ik, ok, sk) in zip(ring_xy(t, orient, n_line), el(cur)):
            ik = (ik+sh) % n_line
            d = np.gradient(r, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True)+1e-12
            p = r[ik]+ok[:, None]*np.stack([d[ik, 1], -d[ik, 0]], 1)
            rr, cc = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
            m = (rr >= 0) & (rr < sz_y) & (cc >= 0) & (cc < sz_x)
            np.add.at(img, (rr[m], cc[m]), sk[m])
        v[t] = gaussian_filter(img, dot_sigma)*2*np.pi*dot_sigma**2
    return v


def to8(v, lo, hi):
    return (np.clip((v-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)


def save(name, vH, vV):
    gap = np.full((vH.shape[0], sz_y, 12), 255, np.uint8)
    sym = lambda v: np.percentile(np.abs(v), 99.5)                     # each half: symmetric range, background mid-grey
    frames = np.concatenate([to8(vH, -sym(vH), sym(vH)), gap, to8(vV, -sym(vV), sym(vV))], 2)
    frames = np.repeat(np.repeat(frames, 2, 1), 2, 2)                  # 2x up-sampled for viewing
    imageio.mimsave(VD+name+'.mp4', list(frames), fps=30, codec='libx264', quality=8, macro_block_size=1)


if __name__ == '__main__':
    base = {o: two_ring_video(o) for o in ('H', 'V')}
    if 'nocomp' not in sys.argv and 'texture' not in sys.argv:
        save('original', base['H'], base['V'])
        save('notch', SF.notched(base['H']), SF.notched(base['V']))
        save('dots', two_ring_video('H', dots=True), two_ring_video('V', dots=True))
    WIG = [.5, 1., 2.]
    if 'texture' in sys.argv:                                          # random-dot rims: vertical rings only, and both
        if 'lifeonly' not in sys.argv:
            tH, tV = two_ring_texture_video('H'), two_ring_texture_video('V')
            save('textureV', base['H'], tV); save('texture_both', tH, tV)
            save('streak_both', two_ring_texture_video('H', streak=16.), two_ring_texture_video('V', streak=16.))
        save('texture_life8_both', two_ring_texture_video('H', lifetime=8), two_ring_texture_video('V', lifetime=8))
        print('texture done', flush=True)
    elif 'nocomp' not in sys.argv or 'wiggle' in sys.argv:
        for a in WIG:                                                  # wiggle on the vertically rotating rings only
            save(f'wiggleV_a{a:g}', base['H'], two_ring_video('V', wiggle=a)); print('wiggle', a, flush=True)
    if 'nocomp' not in sys.argv and 'texture' not in sys.argv:
        for b in LEVELS:
            save(f'compensated_b{b:g}', SF.compensated(base['H'], b), SF.compensated(base['V'], b)); print('compensated', b, flush=True)
    names = {'original': 'Original two rings', 'compensated': 'Compensating filter', 'notch': 'Orientation notch', 'dots': 'Dotted rings'}
    names.update({f'wiggleV_a{a:g}': f'Wiggle {a:g} px on the vertical rings only' for a in WIG})
    names.update({'textureV': 'Random-dot rims on the vertical rings only', 'texture_both': 'Random-dot rims on both',
                  'streak_both': 'Control: random streak rims (16 px, parallel to the rim) on both',
                  'texture_life8_both': 'Random-dot rims, 8-frame dot lifetime (limited tracking), on both'})
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Two rings, filtered</title>
<style>body{font-family:Arial,sans-serif;margin:16px;color:#222;background:#fff;max-width:1100px}
video{width:100%;max-width:1060px;background:#000}.ctl{font-size:17px;margin:10px 0}label{margin-right:18px}
input[type=range]{width:300px;vertical-align:middle}</style></head><body>
<h2>Experimental two-ring stimulus: horizontal (left) and vertical (right) rotation</h2>
<p>Does a stimulus filter stop the wobble percept? Compensating filter: the video multiplied in the Fourier domain by
&radic;(S<sub>iso</sub>/S<sub>aniso</sub>), the population sensitivity of an isotropic over an anisotropic cortex with width anisotropy
&beta; &times; cat (one band, f<sub>0</sub> = 0.12 cycles/px). Orientation notch: the most anisotropic orientations removed. Dotted rings:
isotropic dots on the rings. Random-dot rims: each rim an 8 px band of random dots moving with the ring; control: the same texture with every dot
stretched along the rim into a 16 px streak (orientation energy of the contour, so the aperture problem remains). Wiggle: the vertical rings' contours displaced by A sin(24&theta;) along their normal, moving with the rings (horizontal rings unchanged).
Videos loop; contrast is normalised per setting.</p>
<div class="ctl"><label>Stimulus <select id="stim"></select></label>
<label id="bl">Compensation designed for &beta; = <span id="bv"></span> &times; cat <input id="b" type="range" min="0" max="__NB__" step="1" value="4"></label></div>
<video id="vid" autoplay loop muted playsinline controls></video>
<script>
const N = __NAMES__, L = __LEVELS__; const sel = document.getElementById('stim'), sl = document.getElementById('b'), v = document.getElementById('vid');
for (const [k, n] of Object.entries(N)) { const o = document.createElement('option'); o.value = k; o.textContent = n; sel.appendChild(o); }
function upd() { const s = sel.value, b = L[+sl.value]; document.getElementById('bv').textContent = b.toFixed(1);
  document.getElementById('bl').style.opacity = s === 'compensated' ? 1 : .35; sl.disabled = s !== 'compensated';
  const src = 'stimfilter_videos/'+(s === 'compensated' ? 'compensated_b'+(Math.round(b*10)/10) : s)+'.mp4';
  if (!v.src.endsWith(src)) { v.src = src; v.play(); } }
sel.onchange = upd; sl.oninput = upd; upd();
</script></body></html>"""
    html = html.replace('__NAMES__', json.dumps(names)).replace('__LEVELS__', json.dumps([float(x) for x in LEVELS])).replace('__NB__', str(len(LEVELS)-1))
    open(OUT+'R2_86_figS_two_ring_filters.html', 'w').write(html); print('done')
