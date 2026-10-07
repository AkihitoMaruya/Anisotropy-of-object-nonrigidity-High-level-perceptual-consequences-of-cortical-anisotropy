#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orientation spread of the random-dot rim: each texture element is a dot stretched along the ring (Gaussian arc,
SD `streak` px; Revision_R2_stimulus_filters.texture_video), each element living LIFE frames and then redrawn at a
random new place (staggered), so no element can be tracked through the rotation. Long streaks keep the rim's orientation energy (aperture problem kept); round dots (streak 0) spread it over
all orientations. Best k (direction matching) and match quality of the width-anisotropy cortex (one band, equal
numbers, no smoothing) for every streak length x beta, horizontal and vertical rotation.
Usage: python Revision_R2_texture_spread_sweep.py [minutes]  (time budget; run again to continue). New analysis."""
import os, sys, time
os.environ.setdefault('K_FIT_TRIM', '10')
import numpy as np
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
import Revision_R2_stimulus_filters as SF
import Revision_R2_heeger_temperature_sweep as TS
from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
from Revision_R2_two_rings_ME import sz_x, sz_y
D, P = TS.D, TS.P
LIFE = 8                                       # elements live 8 frames (about 270 ms), then reappear elsewhere: no long-term tracking
STREAKS = [16.0, 8.0, 4.0, 2.0, 1.0, 0.0]
ORIENTS = sys.argv[2:] or ['V', 'H']
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
DOT_SIGMA = 1.0


def spread_deg(streak):
    """Orientation spread of one element's spectrum: half-angle (deg) of its Gaussian spectrum at half height,
    atan(sigma_across/sigma_along) of the element (dot blur SD 1 px across, sqrt(streak^2+1) along)."""
    return np.degrees(np.arctan(DOT_SIGMA/np.hypot(streak, DOT_SIGMA)))


def cache(o):
    return P.A.data+f'heeger_spread_life{LIFE}_{o}.npz'


def results():
    out = {}
    for o in ('H', 'V'):
        F = dict(np.load(cache(o))) if os.path.exists(cache(o)) else {}
        orig = np.load(P.A.data+f'heeger_stimfilter_{o}.npz')
        for b in BETAS:
            k, s = D.fit3(*D.flow_phys(orig[f'original_b{b}']), o); out[(o, 'orig', b)] = (float(k), float(s.max()))
            for st in STREAKS:
                if f's{st}_b{b}' in F:
                    k, s = D.fit3(*D.flow_phys(F[f's{st}_b{b}']), o); out[(o, st, b)] = (float(k), float(s.max()))
    return out


if __name__ == '__main__':
    budget = float(sys.argv[1])*60 if len(sys.argv) > 1 else 1e9; T0 = time.time()
    for st in STREAKS:
        for o in ORIENTS:
            F = dict(np.load(cache(o))) if os.path.exists(cache(o)) else {}
            todo = [b for b in BETAS if f's{st}_b{b}' not in F]
            if not todo:
                continue
            L, _, _ = ring_points(o); L = L[FRAMES]
            pts = (np.repeat(FRAMES, NP), np.clip(np.round(L[..., 0]), 0, sz_y-1).astype(int).ravel(),
                   np.clip(np.round(L[..., 1]), 0, sz_x-1).astype(int).ravel())
            v = SF.texture_video(o, streak=st, lifetime=LIFE)
            for b in todo:
                if time.time()-T0 > budget:
                    print('time budget reached: run again to continue'); sys.exit()
                t0 = time.time(); bk = SF.bank(b)
                F[f's{st}_b{b}'] = bk.flow(bk.energies(v, pts)).reshape(len(FRAMES), NP, 2); np.savez(cache(o), **F)
                print(f'  {o} streak {st:g} beta {b:g}: {(time.time()-t0)/60:.1f} min', flush=True)
    R = results()
    for o in ('H', 'V'):
        print('Horizontal' if o == 'H' else 'Vertical')
        for st in ['orig']+STREAKS:
            print(f'  {st if st == "orig" else f"streak {st:g} ({spread_deg(st):.0f} deg)"}: '
                  + ' '.join(f'{R[(o, st, b)][0]:.2f}({R[(o, st, b)][1]:.2f})' for b in BETAS if (o, st, b) in R))
    print('done')
