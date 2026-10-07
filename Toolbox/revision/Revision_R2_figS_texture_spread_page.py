#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive matching page: the original two rings (horizontal vs vertical rotation) on top; below, the
horizontal and the vertical rings, each with its own knob, which goes from the smooth rings (no texture) to
limited-lifetime random-dot rims (8 frames) whose orientation power moves in steps of about sqrt(2) from concentrated
along the rim (16 px streaks) to scattered over all orientations (round dots) (videos: Revision_R2_figS_texture_spread_videos.py, tex_{H|V}_s*.mp4).
The viewer compares the two rotations at any pair of settings; the settings are shown and kept in the browser.
No model predictions. Videos at native resolution.
Output: Images/Revision_R2/R2_86_figS_texture_spread.html (videos embedded), or with a path argument the page body
plus the videos in v/ next to it, for publishing. New analysis."""
import os, sys, json, base64, tempfile
import imageio.v2 as imageio
current_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Codes/
sys.path.append(current_folder+'/Toolbox/'); sys.path.append(current_folder+'/Toolbox/revision/')
from Revision_R2_figS_texture_spread_videos import PAGE_STREAKS as STREAKS, LIFE, one_path
OUT = current_folder+'/Toolbox/Data/work/'
DOT_SIGMA = 1.0


ART = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.html') else None                    # publishing: page body + videos as files in v/


def embed(name, key):
    """The mp4 at native resolution (the saved videos are 2x up-sampled): base64 for the local page, or (publishing)
    a file v/{key}.mp4 next to the page body, referenced by its relative path."""
    r = imageio.get_reader(name if name.endswith('.mp4') else OUT+f'stimfilter_videos/{name}.mp4'); fr = [r.get_data(i)[::2, ::2] for i in range(r.count_frames())]
    p = tempfile.mktemp(suffix='.mp4'); imageio.mimsave(p, fr, fps=30, codec='libx264', quality=7, macro_block_size=1)
    b = open(p, 'rb').read(); os.remove(p)
    if ART:
        os.makedirs(os.path.dirname(ART)+'/v', exist_ok=True); open(os.path.dirname(ART)+f'/v/{key}.mp4', 'wb').write(b)
        return f'v/{key}.mp4'
    return 'data:video/mp4;base64,'+base64.b64encode(b).decode()


import numpy as np
VIDS = {'original': embed('original', 'original')}
LEVELS = ['smooth']+list(STREAKS)
VIDS.update({f'{o}{i}': embed(one_path(o, s), f'{o}{i}') for o in ('H', 'V') for i, s in enumerate(LEVELS)})
LABELS = ['smooth contour (no texture)']+[f'streaks, SD {s:g} px (orientation spread ±{np.degrees(np.arctan(DOT_SIGMA/np.hypot(s, DOT_SIGMA))):.0f}°)'
          if s else 'round dots (all orientations)' for s in STREAKS]

html = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Random-dot spread match</title>
<style>
/* single column: reference video, adjustable video, knob and setting read-out */
:root{--bg:#fbfbfa;--fg:#1f2328;--mut:#5f6670;--line:#d9dcdf;--grid:#eceef0;--knob:#c9cdd2;--acc:#b03a2e}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#16181b;--fg:#e6e8ea;--mut:#9aa1a9;--line:#3a3f45;--grid:#2a2e33;--knob:#4a5058;--acc:#f07a6c;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#16181b;--fg:#e6e8ea;--mut:#9aa1a9;--line:#3a3f45;--grid:#2a2e33;--knob:#4a5058;--acc:#f07a6c;color-scheme:dark}
body{font-family:Arial,Helvetica,sans-serif;margin:0 auto;padding:16px;color:var(--fg);background:var(--bg);max-width:1100px}
video{width:100%;max-width:1060px;background:#000;display:block}
h2{margin:18px 0 6px;text-wrap:balance}p{line-height:1.45;margin:6px 0;max-width:72ch}
.lab{display:flex;justify-content:space-around;max-width:1060px;font-size:16px;color:var(--mut);margin:4px 0}
.pair{display:flex;gap:12px;max-width:1060px;flex-wrap:wrap}.pair>div{flex:1 1 300px;min-width:0}
.pair h3{margin:6px 0;font-size:16px;font-weight:normal;color:var(--mut)}
.pair .ctl>div{width:100%}.pair input[type=range],.pair .ends{width:100%;max-width:none}
.kv{font-size:15px;font-weight:bold;color:var(--acc);font-variant-numeric:tabular-nums;min-height:2.6em}
.ctl{font-size:16px;margin:12px 0;display:flex;flex-wrap:wrap;gap:8px 22px;align-items:center}
input[type=range]{width:360px;max-width:80vw;vertical-align:middle}
.ends{display:flex;justify-content:space-between;width:360px;max-width:80vw;font-size:13px;color:var(--mut)}
#kv{font-size:18px;font-weight:bold;color:var(--acc);font-variant-numeric:tabular-nums}

</style></head><body>
<h2>Original two rings</h2>
<div class="lab"><span>horizontal rotation</span><span>vertical rotation</span></div>
<video id="orig" autoplay loop muted playsinline></video>
<h2>Rings with limited-lifetime random-dot rims</h2>
<p>Each knob starts at the original smooth rings. The next steps replace the rim by an 8 px band of random dots.
Each dot lives 8 frames (about 270 ms) and then reappears at a random new place, so no dot can be followed through
the rotation. At the first textured step every dot is stretched along the rim into a long streak, so the
orientation power lies along the rim as for the smooth contour. Toward the right the streaks shorten to round dots,
which spread the power over all orientations.</p>
<p>Set each knob and compare how rigid the horizontal (left) and vertical (right) rings look, for example: at which
setting does each pair start to look rigidly connected?</p>
<div class="pair">
<div><h3>horizontal rotation</h3><video id="vidH" autoplay loop muted playsinline></video>
<div class="ctl"><div><input id="knobH" type="range" min="0" max="__NS__" step="1" value="0" aria-label="orientation spread, horizontal rotation">
<div class="ends"><span>smooth</span><span>round dots</span></div></div></div>
<div>Setting <span id="posH"></span>: <div class="kv" id="kvH"></div></div></div>
<div><h3>vertical rotation</h3><video id="vidV" autoplay loop muted playsinline></video>
<div class="ctl"><div><input id="knobV" type="range" min="0" max="__NS__" step="1" value="0" aria-label="orientation spread, vertical rotation">
<div class="ends"><span>smooth</span><span>round dots</span></div></div></div>
<div>Setting <span id="posV"></span>: <div class="kv" id="kvV"></div></div></div>
</div>

<script>
const VIDS = __VIDS__, LAB = __LABELS__, URL_ = {};
function vurl(n) { return VIDS[n]; }
document.getElementById('orig').src = vurl('original');
const V_ = {H: document.getElementById('vidH'), V: document.getElementById('vidV')};
function upd(o) { const knob = document.getElementById('knob'+o), vid = V_[o], other = V_[o === 'H' ? 'V' : 'H'];
  const i = +knob.value; document.getElementById('kv'+o).textContent = LAB[i];
  document.getElementById('pos'+o).textContent = (i+1)+' of '+LAB.length;
  try { localStorage.setItem('knob13'+o, i); } catch (e) {}
  const src = vurl(o+i); if (vid.getAttribute('src') !== src) { vid.src = src;      // keep the two rotations in step
    vid.addEventListener('loadedmetadata', () => { vid.currentTime = other.currentTime || 0; vid.play(); }, {once: true}); } }
for (const o of ['H', 'V']) { const k = document.getElementById('knob'+o);
  try { const s = localStorage.getItem('knob13'+o); if (s !== null) k.value = s; } catch (e) {}
  k.oninput = () => upd(o); upd(o); }
</script></body></html>"""
html = html.replace('__VIDS__', json.dumps(VIDS)).replace('__LABELS__', json.dumps(LABELS)).replace('__NS__', str(len(LEVELS)-1))
if not ART:
    open(OUT+'R2_86_figS_texture_spread.html', 'w').write(html)
if ART:                                                             # page body only (publishing wraps it in a document)
    body = html.split('<head>', 1)[1].replace('</head><body>', '').replace('</body></html>', '')
    body = body.replace('<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n', '')
    open(ART, 'w').write(body)
print('done')
