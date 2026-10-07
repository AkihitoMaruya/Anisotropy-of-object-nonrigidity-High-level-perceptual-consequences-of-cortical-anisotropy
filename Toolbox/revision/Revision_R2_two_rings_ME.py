#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion-energy optic flow and differential invariants for the bottom and top ring
separately (Reviewer 2, Domini et al. 1997), rigid rotation only.

Same pipeline as Figure5C_Horizontal_Rotation.py / Figure5C_Vertical_Rotation.py at 0% stretch;
the only change is the ring tilt: bottom ring +phi (the ring used in Figures 5-6), top ring -phi.

Usage: python Revision_R2_two_rings_ME.py [ring] [orient] [cortex]
    ring in {bottom, top}, orient in {H, V}, cortex in {uniform, aniso}; omitted = all.
Saves Toolbox/Data/Rings/{orient}_{div,curl,def}_{cortex}_{ring}.npy (+ raw U, V).
"""

import os
import sys
import time
import numpy as np
import torch

current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
toolbox_path = current_folder+'/Toolbox/'
sys.path.append(toolbox_path)
from Vis_vec_field import make_rotating_stim
from Compute_optic_flow_from_3D_gabor import Heeger_pyr_flow

save_path = toolbox_path+'Data/Rings/'
tmp_path = current_folder+'/Images/Revision_R2/me_tmp/'
os.makedirs(save_path, exist_ok=True)
os.makedirs(tmp_path, exist_ok=True)

sz_x = sz_y = 128*2
duration = 128
num_points = 500
Omega = np.linspace(0, 180, duration, endpoint=False)*np.pi/180
theta = np.linspace(0, 2*np.pi, num_points)
TILT = {'bottom': 30*np.pi/180, 'top': -30*np.pi/180}


def rotate(x, y, rot=np.pi/2):
    return np.cos(rot)*x-np.sin(rot)*y, np.sin(rot)*x+np.cos(rot)*y


def stimulus_and_fields(phi, orient):
    """Rigid rotation of one ring (Figure5C, 0% stretch): positions, velocities, and
    the curl / div / def operator fields along the ring."""
    X = np.cos(phi)*np.cos(theta)
    Y = np.sin(phi)*np.cos(theta)
    Z = np.sin(theta)
    pos1 = np.outer(np.cos(Omega), X)+np.outer(np.sin(Omega), Z)
    pos2 = np.outer(np.ones_like(Omega), Y)
    vec1 = -np.outer(np.sin(Omega), X)+np.outer(np.cos(Omega), Z)
    vec2 = np.zeros_like(vec1)
    curl1 = -(-np.outer(np.cos(Omega)*np.cos(phi), np.sin(theta))+np.outer(np.sin(Omega), np.cos(theta)))
    curl2 = -(-np.outer(np.sin(phi)*np.ones_like(Omega), np.sin(theta)))
    div1 = np.outer(np.sin(phi)*np.ones_like(Omega), np.sin(theta))
    div2 = -np.outer(np.cos(Omega)*np.cos(phi), np.sin(theta))+np.outer(np.sin(Omega), np.cos(theta))
    if orient == 'V':
        pos1, pos2 = rotate(pos1, pos2)
        vec1, vec2 = rotate(vec1, vec2)
        curl1, curl2 = rotate(curl1, curl2)
        div1, div2 = rotate(div1, div2)
    fields = dict(div=(div1, div2), curl=(curl1, curl2), def1=(-curl1, curl2), def2=(-div1, div2))
    return pos1, pos2, vec1, vec2, fields


def invariants(U, V, fields, num_cut=5):
    I = {g: np.sum(f[0]*U+f[1]*V, axis=1)[num_cut:-num_cut] for g, f in fields.items()}
    out = dict(div=I['div'], curl=I['curl'], def_=np.hypot(I['def1'], I['def2']))
    norm = max(np.abs(v).max() for v in out.values())
    return {g: v/norm for g, v in out.items()}


def run(ring, orient, cortex):
    t0 = time.time()
    pos1, pos2, vec1, vec2, fields = stimulus_and_fields(TILT[ring], orient)
    Video, ex_idx = make_rotating_stim(pos1, pos2, vec1, -vec2, name=f'{orient}_{ring}_ring', sz_x=sz_x, sz_y=sz_y, scale=.01)
    ex_idx = np.hstack(ex_idx)   # make_rotating_stim only stacks it when Vis==1
    VideoR = torch.tensor(Video).reshape(1, duration, sz_y, sz_x).type(torch.float32)
    kw = dict(sname=tmp_path+f'{orient}_{cortex}_{ring}', vfile=tmp_path, name=f'{orient}_{cortex}_{ring}',
              scale=.5, fps=10, smoothness=False, alpha=100, ex_idx=ex_idx, num_scales=2, Vis=False)
    if cortex == 'aniso':
        num_cells = np.load(toolbox_path+'Data/anisotropy_pyr_num_cells_dir.npy')
        kw['num_cells'] = num_cells/num_cells.mean()
    flow = Heeger_pyr_flow(UorA='U_pyr' if cortex == 'uniform' else 'A_pyr', **kw)
    u_hat, v_hat, _, _ = flow.forward(VideoR)
    U = u_hat[ex_idx[0, :], ex_idx[1, :], ex_idx[2, :]].reshape(duration, -1)
    V = -v_hat[ex_idx[0, :], ex_idx[1, :], ex_idx[2, :]].reshape(duration, -1)
    inv = invariants(U, V, fields)
    for g, name in (('div', 'div'), ('curl', 'curl'), ('def_', 'def')):
        np.save(save_path+f'{orient}_{name}_{cortex}_{ring}.npy', inv[g])
    np.save(save_path+f'{orient}_U_{cortex}_{ring}.npy', U)
    np.save(save_path+f'{orient}_V_{cortex}_{ring}.npy', V)
    print(f'{ring} {orient} {cortex}: done in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    rings = [sys.argv[1]] if len(sys.argv) > 1 else ['bottom', 'top']
    orients = [sys.argv[2]] if len(sys.argv) > 2 else ['H', 'V']
    cortices = [sys.argv[3]] if len(sys.argv) > 3 else ['uniform', 'aniso']
    for ring in rings:
        for orient in orients:
            for cortex in cortices:
                run(ring, orient, cortex)
