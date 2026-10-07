# Anisotropy of object nonrigidity — figure code

Code that makes every figure of the paper (Maruya & Zaidi) into `figures_paper/`.
All paths are relative to this folder, so it runs from any location (local copy, Google Drive, Colab).

## Quick start

```bash
pip install -r requirements.txt
python run_all.py              # all figures (Fig 1-6, Fig S1-S5)
python run_all.py Fig5 FigS7   # selected figures
```

On Google Colab, after mounting Drive:

```python
%cd /content/drive/MyDrive/<path to>/Codes
!pip install -r requirements.txt
!python run_all.py
```

## Files

| Script | Output (in `figures_paper/`; videos in `figures_paper/videos/`) |
|---|---|
| `Fig1.py` | `Fig1.pdf`, videos `Fig1A-H.mp4` (rotating rings) |
| `Fig2.py` | `Fig2` (shape anisotropy) |
| `Fig3.py` | `Fig3` (cortical anisotropy: tuning, decoded angles) |
| `Fig4.py` | `Fig4`, video `Fig4A.mp4` (image vs physical stretch, psychophysics) |
| `Fig5.py` | `Fig5`, videos `Fig5C-E.mp4` (optic flow of isotropic / anisotropic cortex, template matching) |
| `Fig6.py` | `Fig6` (differential invariants, Def/Curl fits, stretch) |
| `FigS1.py`, `FigS2.py` | individual observers (shape matches, nonrigidity judgements) |
| `FigS3.py` | intuition: tuning and ring energy per unit (A-D), Heeger cost at one ring point (E, F), best k vs width / number anisotropy (G, H) |
| `FigS3_filters_interactive.py` | `FigS3_filters.html`: 3-D view of all filters, adjustable width anisotropy, optional ring-video spectrum |
| `FigS4.py` | orientation energy of the rim (dynamic random dots), vertical and horizontal rotation, video `FigS4.mp4` (rows A-D together) |
| `FigS4_interactive.py` | `FigS4_matching_task.html`: interactive task, one texture knob for each rotation (videos embedded) |
| `FigS5.py` | the shape of the object does not change Def / Curl: Div, Curl, Def (contour integrals) and Def / Curl for a circle, ellipse, octagon, star and random outline |
| `FigS6.py` | Def / Curl of rotation to wobbling is unchanged by rotating the image (0, 45, 90 = vertical rotation, 135 deg), for ring tilts 15-60 deg |
| `FigS7.py` | local variance of deformations (Domini et al., 1997) on the two rings |

`Toolbox/` holds the model and helpers:
- `figstyle.py` — common figure format (180 mm wide, Arial 10 pt, PDF, images at 300 dpi); output folder.
- `ring_invariants.py` — closed-form Div, Curl, Def of the rotating / wobbling ring (Supplementary Eqs. S18-S29), used for Figure 6.
- `fig5_model.py` — the revised motion-energy model of Figure 5 (Gaussian Heeger model, cat tuning widths x2.5,
  Horn–Schunck smoothness), used by Fig 5 and Fig 6.
- `experiment_data.py` — loads the psychophysical data (`Toolbox/Data/experiment`).
- `revision/` — model code for the analyses (flows, invariants, stretch, texture rims, deformation variance).
- `Data/experiment` — psychophysical data; `Data/videos` — original videos; `Data/Rings` — cached model results
  (missing caches are recomputed automatically by the model code, which can take hours); `Data/work` — stimulus videos.
