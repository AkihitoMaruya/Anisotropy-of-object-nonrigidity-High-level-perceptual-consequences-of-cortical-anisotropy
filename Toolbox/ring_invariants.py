#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed-form differential invariants of the rotating / wobbling ring (Supplementary Eqs. S18-S20 evaluated
analytically).

Motion (corrected S4/S5/S16): P_k = R_y(W) R_Z(tau) R_y(-k W) P(theta), P(theta) = (cos theta, 0, sin theta), W = omega t,
orthographic projection. With s = sin tau, c = cos tau and psi = theta + k W, the image position and velocity are
    x = c cos W cos psi + sin W sin psi,           y = s cos psi
    Fx = omega [(k - c) sin W cos psi + (1 - kc) cos W sin psi],   Fy = -omega k s sin psi
Both are linear in (cos psi, sin psi), so the flow on the ring is exactly linear, F = J (x, y)^T, with
    J = omega [[(1 - kc) cot W,  (k (1 - s^2 cos^2 W) - c) / (s sin W)],
               [-k s / sin W,     k c cot W                          ]]
and the contour integrals S18-S20 equal A x (invariant of J) with the signed enclosed area A = -pi s sin W
(Green's theorem):
    Div  = J11 + J22 = omega cot W
    Curl = J21 - J12 = omega [c - k (1 + s^2 sin^2 W)] / (s sin W)
    Def1 = J12 + J21 = omega [k (c^2 - s^2 cos^2 W) - c] / (s sin W)      (shear, Figure 6A 'Deformation 1')
    Def2 = J11 - J22 = omega (1 - 2kc) cot W                              (Figure 6A 'Deformation 2')
    Def  = sqrt(Def1^2 + Def2^2)
Vertical rotation is the horizontal image turned by 90 deg (J -> [[J22, -J21], [-J12, J11]]); a horizontal image stretch
by s_x maps J -> S J S^-1 with S = diag(s_x, 1) and the area A -> s_x A."""
import numpy as np

TAU = np.radians(30)                                               # ring tilt (Make_rotating_two_rings.py)


def jacobian(k, W, tau=TAU, omega=1.0):
    """J (..., 2, 2) of the flow on the ring at phase W (rad), wobble weight k (horizontal rotation)."""
    s, c = np.sin(tau), np.cos(tau)
    W = np.asarray(W, float)
    J = np.empty(W.shape+(2, 2))
    J[..., 0, 0] = (1-k*c)/np.tan(W)
    J[..., 0, 1] = (k*(1-s**2*np.cos(W)**2)-c)/(s*np.sin(W))
    J[..., 1, 0] = -k*s/np.sin(W)
    J[..., 1, 1] = k*c/np.tan(W)
    return omega*J


def area(W, tau=TAU):
    """Signed area enclosed by the projected unit ring, A = -pi s sin W."""
    return -np.pi*np.sin(tau)*np.sin(W)


def orient(J, o):
    """'H': as is; 'V': the image turned by 90 deg."""
    if o == 'H':
        return J
    Jv = np.empty_like(J)
    Jv[..., 0, 0], Jv[..., 0, 1], Jv[..., 1, 0], Jv[..., 1, 1] = J[..., 1, 1], -J[..., 1, 0], -J[..., 0, 1], J[..., 0, 0]
    return Jv


def stretch(J, sx):
    """Horizontal image stretch by sx: S J S^-1, S = diag(sx, 1)."""
    Js = J.copy()
    Js[..., 0, 1] *= sx
    Js[..., 1, 0] /= sx
    return Js


def invariants(J):
    """Div, Curl, Def1, Def2, Def of J (per unit area: the contour integrals S18-S20 divided by A)."""
    div = J[..., 0, 0]+J[..., 1, 1]
    curl = J[..., 1, 0]-J[..., 0, 1]
    def1 = J[..., 0, 1]+J[..., 1, 0]
    def2 = J[..., 0, 0]-J[..., 1, 1]
    return div, curl, def1, def2, np.hypot(def1, def2)


def def_curl_deg(J):
    """Def / Curl as the angle atan2(Def, Curl) in degrees (the area cancels)."""
    _, curl, _, _, dfm = invariants(J)
    return np.degrees(np.arctan2(dfm, curl))
