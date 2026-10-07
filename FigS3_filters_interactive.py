#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive companion of Supplementary Figure S3: the motion-energy filters of the Figure 5 model with an adjustable
tuning-width anisotropy (beta x cat tuning widths, 0-5; beta = 2.5 is the model) and an adjustable cell-number anisotropy
(alpha x cat cell numbers, 0-5; alpha = 1 is the model).
Left: all 80 filters (16 directions x 5 speed channels) as half-height spheres in (f_x, f_y, f_t), rotatable;
colour = preferred motion direction; optional overlay of the ring video's power spectrum (horizontal or vertical rotation). Right: direction tuning to a grating drifting at 0.58 px/frame (as
Figure 5A, B), times the cell number; with a ring video selected, also the energy each direction unit
collects from that video over the cycle (black, as Supplementary Figure S3A-D; precomputed for every beta and cached in
Toolbox/Data/Rings/ring_energy_betas.npz). The HTML file embeds everything, so it works offline from any folder.
Output: figures_paper/FigS3_filters.html.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Toolbox'))
import numpy as np
import figstyle as fs
from spherical_gabor_bank import fold, cat_hwhh, THETAS, LAYOUT_C
from anisotropy_heeger import cat_counts

F0, SCALE = .12, .5
BETAS = np.round(np.arange(0, 5.001, .25), 2)
S = {f'{b:g}': list(2*F0*np.sin(fold(cat_hwhh(b))*SCALE/2)/np.sqrt(2*np.log(2))) for b in BETAS}   # filter SD per direction
from Revision_R2_D4G_ring import d4g_ring_video


def spectrum_points(o, n=5000):
    """The n strongest voxels of the ring video's power spectrum within the filters' band (0.06 < |f| < 0.18):
    (f_x, f_y, f_t) and log10 power relative to the maximum."""
    V = d4g_ring_video(o, sigma=1.5); T, H, W = V.shape
    P = np.abs(np.fft.fftshift(np.fft.fftn(V)))**2
    ft = np.fft.fftshift(np.fft.fftfreq(T))[:, None, None]; fy = -np.fft.fftshift(np.fft.fftfreq(H))[None, :, None]
    fx = np.fft.fftshift(np.fft.fftfreq(W))[None, None, :]
    R = np.sqrt(fx**2+fy**2+ft**2); P = np.where((R > .06) & (R < .18), P, 0)
    i = np.argsort(P.ravel())[-n:]; t, y, x = np.unravel_index(i, P.shape)
    lp = np.log10(P.ravel()[i]/P.max())
    return [[round(float(fx[0, 0, a]), 4), round(float(fy[0, b, 0]), 4), round(float(ft[c, 0, 0]), 4), round(float(l), 2)]
            for a, b, c, l in zip(x, y, t, lp)]


DS = np.where(np.asarray(LAYOUT_C) < 0)[0]                            # direction-selective channels


def ring_energy(o):
    """(len(BETAS), 16) energy each direction unit collects from the ring video over the whole cycle: sum over its
    direction-selective channels of |video spectrum|^2 x filter^2 (half-space as the model), without cell numbers
    (as Supplementary Figure S3A-D)."""
    V = d4g_ring_video(o, sigma=1.5); T, H, W = V.shape
    P = np.abs(np.fft.fftshift(np.fft.fftn(V)))**2
    ft = np.fft.fftshift(np.fft.fftfreq(T))[:, None, None]; fy = -np.fft.fftshift(np.fft.fftfreq(H))[None, :, None]
    fx = np.fft.fftshift(np.fft.fftfreq(W))[None, None, :]
    F = np.stack(np.broadcast_arrays(fx, fy, ft), -1).reshape(-1, 3); P = P.ravel()
    out = np.zeros((len(BETAS), 16))
    for i, b in enumerate(BETAS):
        sd = np.array(S[f'{b:g}'])
        for k, th in enumerate(THETAS):
            half = (np.cos(th)*F[:, 0]+np.sin(th)*F[:, 1]) > 0
            for ph in np.asarray(LAYOUT_C)[DS]:
                mu = F0*np.array([np.cos(ph)*np.cos(th), np.cos(ph)*np.sin(th), np.sin(ph)])
                out[i, k] += np.sum(P*np.exp(-np.sum((F-mu)**2, 1)/sd[k]**2)*half)
    return out


EN_CACHE = os.path.join(fs.DATA, 'Rings', 'ring_energy_betas.npz')
if os.path.exists(EN_CACHE) and np.allclose(np.load(EN_CACHE)['betas'], BETAS):
    EN = {o: np.load(EN_CACHE)[o] for o in ('H', 'V')}
else:
    EN = {o: ring_energy(o) for o in ('H', 'V')}; np.savez(EN_CACHE, betas=BETAS, **EN)
def point_data(o):
    """Supplementary Figure S3E/F at one ring point (45 deg phase; the point chosen as in FigS3.py): the ring outline,
    the point, the aperture constraint (tangent, normal), the true rigid (k = 0) and wobble (k = 1) velocities, and the
    energy each of the 80 filters measures there (for every beta, equal numbers; cell numbers are applied in the page)."""
    import Revision_R2_heeger_temperature_sweep as TS
    from Revision_R2_heeger_gaussian_flow import ring_points, FRAMES, NP
    from Revision_R2_two_rings_ME import Omega
    from spherical_gabor_bank import exact_bank_sym
    D_, P_ = TS.D, TS.P
    fi = int(np.argmin(np.abs(np.degrees(Omega[FRAMES])-45))); t = FRAMES[fi]
    Pp = ring_points(o)[0][FRAMES][fi]
    Z = np.load(P_.A.data+f'heeger_width_vs_number_{o}.npz')
    R0 = -np.stack(D_.unit(*D_.tmpl_vel(0, o)), -1)[fi]
    ang = lambda v: np.degrees(np.arccos(np.clip((v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)*R0).sum(1), -1, 1)))
    shift = ang(Z['w2.5'][fi])-ang(Z['w0.0'][fi])
    ip = int(np.argmin(shift)) if o == 'H' else int(np.argmax(shift))
    pt = (t, int(round(Pp[ip, 0])), int(round(Pp[ip, 1])))
    tan = Pp[(ip+1) % NP]-Pp[ip-1]; tan = np.array([tan[1], -tan[0]]); tan /= np.linalg.norm(tan)
    nrm = np.array([-tan[1], tan[0]])
    vel = lambda k: -np.array([x[fi, ip] for x in D_.tmpl_vel(k, o)])*P_.A.dO*P_.A.PX
    vid = d4g_ring_video(o, sigma=1.5); T, H, W = vid.shape
    V = np.fft.fftshift(np.fft.fftn(vid))
    ft = np.fft.fftshift(np.fft.fftfreq(T))[:, None, None]; fr = np.fft.fftshift(np.fft.fftfreq(H))[None, :, None]
    fc = np.fft.fftshift(np.fft.fftfreq(W))[None, None, :]
    VP = (V*np.exp(2j*np.pi*(ft*pt[0]+fr*pt[1]+fc*pt[2]))).ravel()/V.size      # energy at one point, without ifftn
    fx, fy, fz = (np.broadcast_to(a, V.shape).ravel() for a in (fc, -fr, ft))
    th_i = np.repeat(THETAS, len(LAYOUT_C))
    mu = np.array([F0*np.array([np.cos(p)*np.cos(th), np.cos(p)*np.sin(th), np.sin(p)]) for th in THETAS for p in LAYOUT_C])
    half = [(np.cos(th)*fx+np.sin(th)*fy) > 0 for th in THETAS]
    E = np.zeros((len(BETAS), len(mu)))
    for i, b in enumerate(BETAS):
        sd = np.repeat(np.array(S[f'{b:g}']), len(LAYOUT_C))
        for j, m in enumerate(mu):
            G = np.exp(-((fx-m[0])**2+(fy-m[1])**2+(fz-m[2])**2)/(2*sd[j]**2))*half[j//len(LAYOUT_C)]
            E[i, j] = np.abs(np.sum(VP*G))**2
    return dict(outline=Pp, point=np.array(pt[1:]), tan=tan, nrm=nrm, T0=vel(0), T1=vel(1), E=E)


PT_CACHE = os.path.join(fs.DATA, 'Rings', 'ring_point_energy_betas.npz')
if os.path.exists(PT_CACHE) and np.allclose(np.load(PT_CACHE)['betas'], BETAS):
    Zc = np.load(PT_CACHE); PT = {o: {k: Zc[f'{o}_{k}'] for k in ('outline', 'point', 'tan', 'nrm', 'T0', 'T1', 'E')} for o in ('H', 'V')}
else:
    PT = {o: point_data(o) for o in ('H', 'V')}
    np.savez(PT_CACHE, betas=BETAS, **{f'{o}_{k}': v for o in PT for k, v in PT[o].items()})
data = dict(point={o: {k: np.round(np.asarray(v, float), 8).tolist() for k, v in PT[o].items()} for o in ('H', 'V')},
            energy={o: [list(map(float, e)) for e in EN[o]] for o in ('H', 'V')},
            spec={o: spectrum_points(o) for o in ('H', 'V')}, f0=F0, thetas=list(THETAS), phis=list(LAYOUT_C), betas=[float(b) for b in BETAS], s=S,
            num=list(fold(cat_counts(THETAS))))

html = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motion-energy filters</title>
<style>
:root{--bg:#fbfbfa;--fg:#1f2328;--mut:#5f6670;--line:#d9dcdf}
@media (prefers-color-scheme: dark){:root{--bg:#16181b;--fg:#e6e8ea;--mut:#9aa1a9;--line:#3a3f45}}
body{font-family:Arial,Helvetica,sans-serif;margin:0 auto;padding:16px;max-width:1150px;background:var(--bg);color:var(--fg)}
h2{margin:6px 0}p{line-height:1.45;max-width:78ch;margin:6px 0}
.ctl{display:flex;flex-wrap:wrap;gap:10px 26px;align-items:center;margin:12px 0;font-size:16px}
input[type=range]{width:320px;max-width:80vw;vertical-align:middle}
.val{font-weight:bold;font-variant-numeric:tabular-nums}
.row{display:flex;flex-wrap:wrap;gap:18px}.row>div{flex:1 1 440px;min-width:0}
canvas{width:100%;height:auto;background:#fff;border:1px solid var(--line);touch-action:none}
#c1{cursor:grab}
.cap{color:var(--mut);font-size:14px}
</style></head><body>
<h2>Motion-energy filters of the model (Figure 5, Supplementary Figure S3)</h2>
<p>All 80 spatiotemporal filters of the model (16 directions &times; 5 speed channels) in the frequency domain
(f<sub>x</sub>, f<sub>y</sub>, f<sub>t</sub>): each sphere is a filter's half-height surface. The anisotropic cortex scales
the cat V1 tuning widths (Li et al., 2003) by &beta; around their mean: &beta; = 0 is the isotropic cortex, &beta; = 1 the cat
widths, &beta; = 2.5 the model in the paper. Drag to rotate.</p>
<div class="ctl"><label for="b">Width anisotropy &beta; = <span class="val" id="bv"></span> &times; cat</label>
<input id="b" type="range" min="0" max="__NB__" step="1" value="__B0__">
<label for="n">Number anisotropy &alpha; = <span class="val" id="nv"></span> &times; cat</label>
<input id="n" type="range" min="0" max="20" step="1" value="4">
<label><input id="st" type="checkbox" checked> static channels (f<sub>t</sub> = 0)</label>
<label>Ring-video spectrum <select id="sp"><option value="">none</option><option value="H">horizontal rotation</option>
<option value="V">vertical rotation</option></select></label>
<button id="rs">Reset view</button></div>
<div class="row"><div><canvas id="c1" width="640" height="640"></canvas>
<p class="cap">Colour = preferred motion direction of the filter's unit (left red, up purple, right cyan, down green);
static channels (f<sub>t</sub> = 0) paler. Black points (optional): the strongest frequencies of the rotating ring video
(darker = more power, top 4 log units). Only the half-space f<sub>x</sub>cos&theta; + f<sub>y</sub>sin&theta; &gt; 0 of each filter
is used, as in the model.</p></div>
<div><canvas id="c2" width="620" height="420"></canvas>
<p class="cap">Response of each direction unit (its two direction-selective channels) to a grating at the filters'
spatial frequency drifting at 0.58 px/frame in each direction (peak 1, times the cell number), as Figure 5A, B. The cell numbers
are the cat V1 numbers scaled by &alpha; around their mean: &alpha; = 0 equal numbers, &alpha; = 1 the cat numbers (the model).
With a ring video selected, black (right axis): the energy each direction unit collects from that video over the whole
cycle, times the cell number, normalised to the isotropic cortex's maximum (&beta; = 0, &alpha; = 0), as Supplementary Figure S3A-D.</p></div></div>
<div class="row"><div><canvas id="c3" width="620" height="620"></canvas></div>
<div><h3 style="margin:6px 0">Velocity estimate at one point of the ring (as Supplementary Figure S3E, F)</h3>
<p class="cap">For the selected ring video (45&deg; phase; the point is marked on the ring outline, top left): the cost of
each velocity v, &Sigma;<sub>i</sub> (n<sub>i</sub>m<sub>i</sub> &minus; m&#772;<sub>i</sub>R<sub>i</sub>(v)/R&#772;<sub>i</sub>(v))<sup>2</sup> (Eq. S15), from the
energies m<sub>i</sub> the 80 filters measure at that point, the predicted responses R<sub>i</sub>(v) and the cell numbers
n<sub>i</sub>, for the current &beta; and &alpha;. Dashed: the aperture constraint (true normal motion); black and grey arrows:
the true velocities for rotation (k = 0) and wobbling (k = 1); red arrow: the model estimate (minimum cost).
With &alpha; = 0 this is Supplementary Figure S3E (horizontal) or F (vertical).</p>
<p id="est" class="val"></p></div></div>
<script>
const D = __DATA__;
const bsl=document.getElementById('b'), nsl=document.getElementById('n'), sck=document.getElementById('st'), spk=document.getElementById('sp');
function hsv(h,v){const i=Math.floor(h*6),f=h*6-i,q=v*(1-f),t=v*f;return [[v,t,0],[q,v,0],[0,v,t],[0,q,v],[t,0,v],[v,0,q]][i%6].map(x=>Math.round(x*255));}
function colour(a){return hsv(((Math.atan2(-Math.sin(a),-Math.cos(a))/(2*Math.PI))%1+1)%1,.9);}
let yaw=-0.6, pitch=0.45;
function rot(p){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const x=cy*p[0]-sy*p[1], y=sy*p[0]+cy*p[1], z=p[2];             // yaw about f_t
  return [x, cp*y-sp*z, sp*y+cp*z];}                               // pitch about screen x; [screen x, depth, screen y]
function draw3d(){
  const b=D.betas[+bsl.value], S=D.s[String(b)], c=document.getElementById('c1'), g=c.getContext('2d');
  const W=c.width, H=c.height, sc=W/0.5, cx=W/2, cy=H/2;
  g.fillStyle='#fff'; g.fillRect(0,0,W,H);
  const P=p=>{const r=rot(p);return [cx+sc*r[0], cy-sc*r[2], r[1]];};
  const L=.2, axes=[[[-L,0,0],[L,0,0],'f_x'],[[0,-L,0],[0,L,0],'f_y'],[[0,0,-L],[0,0,L],'f_t']];
  const items=[];
  for(let k=0;k<16;k++)for(const ph of D.phis){
    if(ph===0 && !sck.checked) continue;
    const th=D.thetas[k], mu=[D.f0*Math.cos(ph)*Math.cos(th),D.f0*Math.cos(ph)*Math.sin(th),D.f0*Math.sin(ph)];
    const p=P(mu); items.push({x:p[0],y:p[1],d:p[2],r:sc*S[k]*Math.sqrt(2*Math.log(2)),col:colour(th),st:ph===0});}
  if(spk.value){const pts=D.spec[spk.value];for(const [x,y,t,l] of pts){const p=P([x,y,t]);items.push({x:p[0],y:p[1],d:p[2],pt:true,a:Math.max(.08,Math.min(.9,1+l/4))});}}
  items.sort((a,b)=>b.d-a.d);                                       // far (large depth) first
  g.lineWidth=1.2; g.font='15px Arial'; g.fillStyle='#333';
  axes.forEach(([a,b2,lab])=>{const p=P(a),q=P(b2);g.strokeStyle='#999';g.beginPath();g.moveTo(p[0],p[1]);g.lineTo(q[0],q[1]);g.stroke();g.fillText(lab,q[0]+4,q[1]-4);});
  const sa=spk.value?.6:1;
  for(const it of items){if(it.pt){g.fillStyle=`rgba(20,20,20,${it.a})`;g.fillRect(it.x-1.5,it.y-1.5,3,3);continue;}
    const [r,gr,bl]=it.col, a=(it.st?.25:.55)*sa;
    const grd=g.createRadialGradient(it.x-it.r*.35,it.y-it.r*.35,it.r*.1,it.x,it.y,it.r);
    grd.addColorStop(0,`rgba(${Math.min(255,r+90)},${Math.min(255,gr+90)},${Math.min(255,bl+90)},${a})`);
    grd.addColorStop(1,`rgba(${r*.7|0},${gr*.7|0},${bl*.7|0},${a})`);
    g.fillStyle=grd; g.beginPath(); g.arc(it.x,it.y,it.r,0,2*Math.PI); g.fill();}
  // direction key: colour of each preferred motion direction
  const kx=W-78, ky=72, kr=38; g.fillStyle='rgba(255,255,255,.85)'; g.fillRect(kx-74,ky-62,148,146);
  for(let i=0;i<16;i++){const a=i*Math.PI/8, ex=kx+kr*Math.cos(a), ey=ky-kr*Math.sin(a);
    g.strokeStyle=g.fillStyle=`rgb(${colour(a)})`; g.lineWidth=3; g.beginPath(); g.moveTo(kx,ky); g.lineTo(ex,ey); g.stroke();
    const hx=Math.cos(a), hy=-Math.sin(a); g.beginPath(); g.moveTo(ex+hx*7,ey+hy*7); g.lineTo(ex-hy*4,ey+hx*4); g.lineTo(ex+hy*4,ey-hx*4); g.fill();}
  g.fillStyle='#222'; g.font='12px Arial'; g.textAlign='center';
  g.fillText('up',kx,ky-kr-14); g.fillText('down',kx,ky+kr+22); g.textAlign='left'; g.fillText('right',kx+kr+10,ky+4);
  g.textAlign='right'; g.fillText('left',kx-kr-10,ky+4); g.textAlign='center';
  g.font='11px Arial'; g.fillText('preferred direction',kx,ky+kr+38); g.textAlign='left';
}
function draw2d(){
  const b=D.betas[+bsl.value], S=D.s[String(b)], num=nums(nsl.value/4);
  const c2=document.getElementById('c2'),h=c2.getContext('2d');h.fillStyle='#fff';h.fillRect(0,0,c2.width,c2.height);
  const X0=60,Y0=20,WW=c2.width-130,HH=c2.height-80,phiS=Math.PI/6,yMax=Math.max(1.5,1.1*Math.max(...num));   // left axis grows with the cell numbers
  const X=a=>X0+(a+180)/360*WW,Y=v=>Y0+HH*(1-v/yMax);
  h.strokeStyle='#888';h.lineWidth=1;h.beginPath();h.moveTo(X0,Y0);h.lineTo(X0,Y0+HH);h.lineTo(X0+WW,Y0+HH);h.stroke();
  h.fillStyle='#222';h.font='14px Arial';h.textAlign='center';[-180,-90,0,90,180].forEach(a=>h.fillText(a,X(a),Y0+HH+18));
  h.fillText('Motion direction (deg)',X0+WW/2,Y0+HH+40);h.textAlign='right';for(let v=0;v<=yMax+1e-9;v+=(yMax>2.5?1:.5))h.fillText(v,X0-6,Y(v)+5);
  const pk=[];
  for(let k=0;k<16;k++){const th=D.thetas[k],R=[];let mx=0,imx=0;
    const ch=D.phis.filter(p=>p<0).map(ph=>[D.f0*Math.cos(ph)*Math.cos(th),D.f0*Math.cos(ph)*Math.sin(th),D.f0*Math.sin(ph)]),s=S[k];
    for(let d=-180;d<=180;d++){const a=d*Math.PI/180,f=[D.f0*Math.cos(phiS)*Math.cos(a),D.f0*Math.cos(phiS)*Math.sin(a),-D.f0*Math.sin(phiS)];let v=0;
      for(const m of ch){v+=Math.exp(-((f[0]-m[0])**2+(f[1]-m[1])**2+(f[2]-m[2])**2)/(s*s))+Math.exp(-((-f[0]-m[0])**2+(-f[1]-m[1])**2+(-f[2]-m[2])**2)/(s*s));}
      R.push(v);if(v>mx){mx=v;imx=d;}}
    pk.push(imx);
    h.strokeStyle=`rgb(${colour(th)})`;h.lineWidth=2;h.beginPath();R.forEach((v,i)=>{const x=X(i-180),y=Y(num[k]*v/mx);i?h.lineTo(x,y):h.moveTo(x,y);});h.stroke();}
  if(spk.value){                                                   // ring energy per unit (black, right axis)
    const o=spk.value,Eb=D.energy[o],iso=Math.max(...Eb[0]);       // normalised to the isotropic cortex's maximum
    const n5=nums(5);let top=0;Eb.forEach(e=>e.forEach((v,k)=>{top=Math.max(top,v*Math.max(1,n5[k])/iso);}));const y2=1.1*top;
    const Y2=v=>Y0+HH*(1-v/y2),E=Eb[+bsl.value].map((v,k)=>v*num[k]/iso);
    const idx=[...Array(16).keys()].sort((a,b)=>pk[a]-pk[b]),xs=idx.map(k=>pk[k]),es=idx.map(k=>E[k]);
    const xw=[xs[15]-360,...xs,xs[0]+360],ew=[es[15],...es,es[0]];
    h.save();h.beginPath();h.rect(X0,Y0-2,WW,HH+4);h.clip();
    h.strokeStyle='#000';h.lineWidth=2.4;h.beginPath();xw.forEach((x,i)=>{const px=X(x),py=Y2(ew[i]);i?h.lineTo(px,py):h.moveTo(px,py);});h.stroke();
    h.fillStyle='#000';xs.forEach((x,i)=>{h.beginPath();h.arc(X(x),Y2(es[i]),3.5,0,2*Math.PI);h.fill();});h.restore();
    h.strokeStyle='#888';h.lineWidth=1;h.beginPath();h.moveTo(X0+WW,Y0);h.lineTo(X0+WW,Y0+HH);h.stroke();
    h.fillStyle='#222';h.font='14px Arial';h.textAlign='left';const st=y2>2?1:.5;
    for(let v=0;v<=y2;v+=st)h.fillText(v.toFixed(st<1?1:0),X0+WW+6,Y2(v)+5);
    h.save();h.translate(X0+WW+44,Y0+HH/2);h.rotate(-Math.PI/2);h.textAlign='center';h.fillText('Ring energy (norm.)',0,0);h.restore();}
}
function nums(a){return D.num.map(n=>1+a*(n-1));}                 // cat cell numbers (mean 1) scaled by alpha around the mean
const VIR=[[253,231,37],[94,201,98],[33,145,140],[59,82,139],[68,1,84]];   // viridis, reversed: low cost = yellow
function vir(t){t=Math.max(0,Math.min(1,t))*(VIR.length-1);const i=Math.min(VIR.length-2,Math.floor(t)),f=t-i;return VIR[i].map((c,j)=>Math.round(c+(VIR[i+1][j]-c)*f));}
function costGrid(o,vx,vy){                                   // Heeger cost (Eq. S15, 'raw' normalisation) for velocities vx, vy
  const b=+bsl.value,S=D.s[String(D.betas[b])],E=D.point[o].E[b],num=nums(nsl.value/4),NS=D.phis.length,out=new Float64Array(vx.length);
  const mu=[];for(let k=0;k<16;k++)for(const p of D.phis)mu.push([D.f0*Math.cos(p)*Math.cos(D.thetas[k]),D.f0*Math.cos(p)*Math.sin(D.thetas[k]),D.f0*Math.sin(p)]);
  for(let q=0;q<vx.length;q++){const nr=Math.hypot(vx[q],vy[q],1),n=[vx[q]/nr,vy[q]/nr,1/nr],R=new Float64Array(80);
    for(let i=0;i<80;i++){const s2=S[Math.floor(i/NS)]**2,d=n[0]*mu[i][0]+n[1]*mu[i][1]+n[2]*mu[i][2];
      R[i]=Math.pow(2*Math.PI,1.5)*Math.pow(s2/2,1.5)*Math.exp(-d*d/s2)/Math.sqrt(Math.PI*s2)/nr;}
    let c=0;for(let a=0;a<8;a++){const g=[];for(const k of [a,a+8])for(let j=0;j<NS;j++)g.push(k*NS+j);   // filters sharing a spatial axis
      let Mb=0,Rb=0;for(const i of g){Mb+=E[i];Rb+=R[i];}
      for(const i of g){const r=num[Math.floor(i/NS)]*E[i]-Mb*R[i]/Rb;c+=r*r;}}
    out[q]=c;}
  return out;}
function arrow(h,x0,y0,x1,y1,col,w){h.strokeStyle=h.fillStyle=col;h.lineWidth=w;h.beginPath();h.moveTo(x0,y0);h.lineTo(x1,y1);h.stroke();
  const a=Math.atan2(y1-y0,x1-x0);h.beginPath();h.moveTo(x1,y1);h.lineTo(x1-12*Math.cos(a-.4),y1-12*Math.sin(a-.4));h.lineTo(x1-12*Math.cos(a+.4),y1-12*Math.sin(a+.4));h.fill();}
function draw3e(){
  const c3=document.getElementById('c3'),h=c3.getContext('2d'),o=spk.value;h.fillStyle='#fff';h.fillRect(0,0,c3.width,c3.height);
  if(!o){h.fillStyle='#555';h.font='16px Arial';h.textAlign='center';h.fillText('Select a ring video above',c3.width/2,c3.height/2);document.getElementById('est').textContent='';return;}
  const pd=D.point[o],VM=o==='H'?2.5:0.6,N=121,X0=70,Y0=20,SZ=c3.width-100;
  const g=[...Array(N).keys()].map(i=>-VM+2*VM*i/(N-1)),vx=[],vy=[];for(const y of g)for(const x of g){vx.push(x);vy.push(y);}
  const C=costGrid(o,vx,vy);
  const G2=[...Array(81).keys()].map(i=>-4+.1*i),fx=[],fy=[];for(const y of G2)for(const x of G2){fx.push(x);fy.push(y);}   // estimate: minimum over +-4 px/frame
  const CF=costGrid(o,fx,fy);let im=0;for(let i=1;i<CF.length;i++)if(CF[i]<CF[im])im=i;const est=[fx[im],fy[im]];
  const sorted=Array.from(C).sort((a,b)=>a-b),cmin=sorted[0],c5=sorted[Math.floor(.05*sorted.length)];
  const cell=SZ/N;for(let j=0;j<N;j++)for(let i=0;i<N;i++){const v=(C[j*N+i]-cmin)/Math.max(c5-cmin,1e-30);h.fillStyle=`rgb(${vir(v)})`;h.fillRect(X0+i*cell,Y0+(N-1-j)*cell,cell+1,cell+1);}
  const P=(x,y)=>[X0+(x+VM)/(2*VM)*SZ,Y0+(1-(y+VM)/(2*VM))*SZ];
  h.save();h.beginPath();h.rect(X0,Y0,SZ,SZ);h.clip();
  const sn=pd.T0[0]*pd.nrm[0]+pd.T0[1]*pd.nrm[1],a=P(sn*pd.nrm[0]-6*pd.tan[0],sn*pd.nrm[1]-6*pd.tan[1]),bq=P(sn*pd.nrm[0]+6*pd.tan[0],sn*pd.nrm[1]+6*pd.tan[1]);
  h.setLineDash([8,6]);h.strokeStyle='#fff';h.lineWidth=2;h.beginPath();h.moveTo(...a);h.lineTo(...bq);h.stroke();h.setLineDash([]);
  const cut=v=>{const L=Math.hypot(...v);return L<=.92*VM?v:v.map(z=>z/L*.92*VM);};
  const o0=P(0,0);arrow(h,...o0,...P(...cut(pd.T0)),'#000',3);arrow(h,...o0,...P(...cut(pd.T1)),'#999',3);arrow(h,...o0,...P(...cut(est)),'#d00',3.5);
  // ring outline and the point (inset)
  const ol=pd.outline,rs=ol.map(p=>p[0]),cs=ol.map(p=>p[1]),r0=Math.min(...rs),r1=Math.max(...rs),c0=Math.min(...cs),c1=Math.max(...cs),sc=110/Math.max(r1-r0,c1-c0);
  h.fillStyle='rgba(255,255,255,.85)';h.fillRect(X0+4,Y0+4,130,130);h.strokeStyle='#000';h.lineWidth=2;h.beginPath();
  ol.forEach((p,i)=>{const x=X0+12+(p[1]-c0)*sc,y=Y0+12+(p[0]-r0)*sc;i?h.lineTo(x,y):h.moveTo(x,y);});h.closePath();h.stroke();
  h.strokeStyle='#d00';h.lineWidth=2.5;h.beginPath();h.arc(X0+12+(pd.point[1]-c0)*sc,Y0+12+(pd.point[0]-r0)*sc,7,0,2*Math.PI);h.stroke();h.restore();
  h.strokeStyle='#888';h.lineWidth=1;h.strokeRect(X0,Y0,SZ,SZ);h.fillStyle='#222';h.font='14px Arial';h.textAlign='center';
  [-VM,0,VM].forEach(v=>{h.fillText(v,P(v,0)[0],Y0+SZ+18);});h.fillText('v_x (px/frame)',X0+SZ/2,Y0+SZ+40);
  h.textAlign='right';[-VM,0,VM].forEach(v=>{h.fillText(v,X0-6,P(0,v)[1]+5);});
  h.save();h.translate(18,Y0+SZ/2);h.rotate(-Math.PI/2);h.textAlign='center';h.fillText('v_y (px/frame)',0,0);h.restore();
  const dot=(u,v)=>(u[0]*v[0]+u[1]*v[1])/Math.max(Math.hypot(...u)*Math.hypot(...v),1e-12),deg=x=>Math.acos(Math.max(-1,Math.min(1,x)))*180/Math.PI;
  document.getElementById('est').textContent=`Estimate ${deg(dot(est,pd.T0)).toFixed(0)}° from rigid rotation, ${deg(dot(est,pd.T1)).toFixed(0)}° from wobbling (${o==='H'?'horizontal':'vertical'} rotation, β = ${D.betas[+bsl.value].toFixed(2)}, α = ${(nsl.value/4).toFixed(2)})`;}
function draw(){document.getElementById('bv').textContent=D.betas[+bsl.value].toFixed(2);document.getElementById('nv').textContent=(nsl.value/4).toFixed(2);draw3d();draw2d();draw3e();}
const c1=document.getElementById('c1');let drag=null;
c1.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];c1.setPointerCapture(e.pointerId);});
c1.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];draw3d();});
c1.addEventListener('pointerup',()=>drag=null);
document.getElementById('rs').onclick=()=>{yaw=-0.6;pitch=0.45;draw3d();};
bsl.oninput=nsl.oninput=sck.onchange=spk.onchange=draw;draw();
</script></body></html>"""
html = (html.replace('__DATA__', json.dumps(data)).replace('__NB__', str(len(BETAS)-1))
        .replace('__B0__', str(int(np.argmin(np.abs(BETAS-2.5))))))
open(os.path.join(fs.OUT, 'FigS3_filters.html'), 'w').write(html)
print('saved', os.path.join(fs.OUT, 'FigS3_filters.html'))
