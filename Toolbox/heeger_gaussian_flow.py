#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heeger (1988) optic flow with 3D Gaussian motion-energy filters, in closed form.

Derivation: heeger_gaussian_model.md (this folder).  Companion to optic_flow_corrected.py,
which does the same for the steerable pyramid's cos^(K-1) filters by numerical
great-circle integration; here the filters are Gaussian, so the white-noise prediction
is analytic.

Filters (frequency domain, cycles/sample, coordinates (fx, fy, ft), fy pointing UP on screen):
    radial x spatial-angle x temporal-angle Gaussians, linearised at their peak:
        mu = f0 (cos phi cos theta, cos phi sin theta, sin phi)
        C  = E diag(sigma_r^2, (f0 cos phi sigma_theta)^2, (f0 sigma_t)^2) E^T,
        E  = [e_r, e_theta, e_a]  (radial, spatial tangent, temporal tangent)
    cut to the half-space cos(theta) fx + sin(theta) fy > 0: one-sided, so |response|^2
    is motion energy (quadrature pair).  sigma_theta may differ per filter.

Expected energy for white noise translating at v (Heeger Eq. 7-8, general covariance):
    the stimulus power is S0 on the plane  ft = -(vx fx + vy fy),  so
    R_i(v) = S0 * integral over (fx, fy) of |G_i|^2
           = S0 (2 pi)^(3/2) sqrt(det S_i) N(0; n.mu_i, n^T S_i n) / |m|,
    S_i = C_i / 2,  m = (vx, vy, 1),  n = m / |m|.

Population: filter i is carried by n_i cells, so the population energy is n_i m_i (measured)
and n_i R_i(v) (predicted); n_i = 1 is Heeger's model.
Combination (Heeger Eq. 10-11): group filters by spatial axis (theta and theta + pi, all
temporal channels), normalise within each group, least squares over v:
    cost(v) = sum_i (n_i m_i - [sum_{j in grp} n_j m_j] n_i R_i(v) / [sum_{j in grp} n_j R_j(v)])^2  Either per point (flow) or pooled over points with the
information-matrix combination of optic_flow_corrected.py (global_velocity).
"""

import numpy as np


def derived_gaussian(theta, phi, sigma_theta, f0, sigma_r, sigma_t):
    """Centre mu and covariance C, (fx, fy_up, ft), cycles/sample."""
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    E = np.array([[cp*ct, -st, -sp*ct],
                  [cp*st, ct, -sp*st],
                  [sp, 0., cp]])                         # columns e_r, e_theta, e_a
    L = np.diag([sigma_r**2, (f0*cp*sigma_theta)**2, (f0*sigma_t)**2])
    return f0*E[:, 0], E@L@E.T


class GaussianBank:
    """thetas: K spatial directions (rad, screen, y up); sigmas: K spatial angular sigmas (rad);
    phis: S temporal angles (rad; 0 static, sign = direction along theta);
    f0, sigma_r (cycles/sample), sigma_t (rad); weights: K numbers of cells per direction
    (default 1), which multiply each filter's energy."""

    def __init__(self, thetas, sigmas, phis, f0=0.12, sigma_r=0.04, sigma_t=np.radians(20), weights=None,
                 normalize='population', k=0.0, norm_exp=1.0, filters='product'):
        self.thetas, self.phis = np.asarray(thetas, float), np.asarray(phis, float)
        self.sigmas = np.broadcast_to(np.asarray(sigmas, float), self.thetas.shape)
        self.K, self.S = len(self.thetas), len(self.phis)
        self.normalize = normalize
        self.k = k                                                   # constant added to the normalising sum Rbar
        self.norm_exp = norm_exp                                     # exponent p on the normalising sum Rbar
        self.filters = filters                                       # 'product': exact 3-Gaussian product for the
        #   measured energies; 'gaussian': the linearised 3D Gaussian. Predictions always use the linearised one.
        self.f0, self.sigma_r, self.sigma_t = f0, sigma_r, sigma_t
        w = np.ones(self.K) if weights is None else np.asarray(weights, float)
        mu, C, th, ww, ph, sg = [], [], [], [], [], []
        for k in range(self.K):
            for j in range(self.S):
                m, c = derived_gaussian(self.thetas[k], self.phis[j], self.sigmas[k], f0, sigma_r, sigma_t)
                mu.append(m); C.append(c); th.append(self.thetas[k]); ww.append(w[k]); ph.append(self.phis[j]); sg.append(self.sigmas[k])
        self.mu, self.C, self.theta_i, self.w = np.array(mu), np.array(C), np.array(th), np.array(ww)
        self.phi_i, self.sigma_i = np.array(ph), np.array(sg)
        axis = np.round(np.mod(np.degrees(self.theta_i), 180), 6)
        self.axes = np.unique(axis)
        self.axis_id = np.searchsorted(self.axes, axis)
        self.groups = [np.where(self.axis_id == a)[0] for a in self.axis_id]

    # ---------------- measured motion energy ----------------
    def masks(self, shape):
        """Yield the one-sided frequency mask of every filter on an fftshifted (T, H, W) grid:
        the exact product radial x spatial-angle x temporal-angle (filters='product') or its
        linearised 3D Gaussian (filters='gaussian')."""
        T, H, W = shape
        ft = np.fft.fftshift(np.fft.fftfreq(T))[:, None, None]
        fy = -np.fft.fftshift(np.fft.fftfreq(H))[None, :, None]      # rows run down, fy up
        fx = np.fft.fftshift(np.fft.fftfreq(W))[None, None, :]
        if self.filters == 'product':
            rho = np.sqrt(fx**2+fy**2+ft**2)
            Rad = np.exp(-(rho-self.f0)**2/(2*self.sigma_r**2))
            ang = np.arctan2(fy, fx)
            for th, ph, sg in zip(self.theta_i, self.phi_i, self.sigma_i):
                d = (ang-th+np.pi) % (2*np.pi)-np.pi
                A = np.exp(-d**2/(2*sg**2))*(np.abs(d) < np.pi/2)
                a = np.arctan2(ft, np.cos(th)*fx+np.sin(th)*fy)
                yield Rad*A*np.exp(-(a-ph)**2/(2*self.sigma_t**2))
            return
        for m, c, th in zip(self.mu, self.C, self.theta_i):
            P = np.linalg.inv(c)
            dx, dy, dt = fx-m[0], fy-m[1], ft-m[2]
            q = (P[0, 0]*dx*dx+P[1, 1]*dy*dy+P[2, 2]*dt*dt+2*(P[0, 1]*dx*dy+P[0, 2]*dx*dt+P[1, 2]*dy*dt))
            yield np.exp(-0.5*q)*((np.cos(th)*fx+np.sin(th)*fy) > 0)

    def energies(self, video, points, video_fft=None, pool=None):
        """video (T, H, W); points = (t, row, col) index arrays -> motion energy (n_points, n_filters).
        video_fft: optional precomputed fftshift(fftn(video)) to reuse across banks.
        pool: None (energy at each point), or a list of (sigma_t, sigma_space) Gaussian pooling widths
            (frames, px); each filter's energy map is blurred (periodic boundary) before sampling, as in
            Heeger (1988). With a list, returns (n_pool, n_points, n_filters); (0, 0) means no pooling."""
        from scipy.ndimage import gaussian_filter
        V = np.fft.fftshift(np.fft.fftn(video)) if video_fft is None else video_fft
        pools = [None] if pool is None else list(pool)
        M = np.zeros((len(pools), len(points[0]), len(self.mu)))
        for i, G in enumerate(self.masks(video.shape)):
            E = (np.abs(np.fft.ifftn(np.fft.ifftshift(V*G)))**2).astype(np.float32)
            for p, pw in enumerate(pools):
                if pw is None or (pw[0] == 0 and pw[1] == 0):
                    M[p, :, i] = E[points]
                else:
                    M[p, :, i] = gaussian_filter(E, (pw[0], pw[1], pw[1]), mode='wrap')[points]
        return M[0] if pool is None else M

    # ---------------- expected energy for moving white noise ----------------
    def predicted(self, vx, vy):
        """R (n_filters, n_v) for white noise translating at (vx, vy_up) px/frame (S0 = 1)."""
        m = np.stack([vx, vy, np.ones_like(vx)], 1)
        nrm = np.linalg.norm(m, axis=1)
        n = m/nrm[:, None]
        R = np.zeros((len(self.mu), len(vx)))
        for i, (mu, c) in enumerate(zip(self.mu, self.C)):
            S = c/2
            s2 = np.einsum('vi,ij,vj->v', n, S, n)
            R[i] = (2*np.pi)**1.5*np.sqrt(np.linalg.det(S))*np.exp(-(n@mu)**2/(2*s2))/np.sqrt(2*np.pi*s2)/nrm
        return R

    def velocity_grid(self, vmax=4.0, dv=0.1):
        g = np.arange(-vmax, vmax+dv/2, dv)
        VX, VY = np.meshgrid(g, g, indexing='xy')
        return g, VX, VY

    # ---------------- Heeger least squares, per point ----------------
    def cost(self, M, R):
        """cost (n_points, n_v) of Heeger Eq. 11.
        normalize='population': measured and predicted energies are both multiplied by the number of
            cells, and normalised by their own sums (the counts cancel within an axis group);
        normalize='raw': the measured population energy n_i m_i is compared with the raw prediction
            mbar_i R_i / Rbar_i, where mbar_i and Rbar_i sum the raw filter responses (counts do not cancel).
        k: constant added to the predicted normalising sum, R_i / (Rbar_i + k) (k = 0: Heeger).
        norm_exp: exponent p, R_i / Rbar_i^p (p = 1: Heeger, 0: no normalisation); for p != 1, R is
            expressed in units of the median Rbar over the velocity grid."""
        if self.normalize == 'population':
            Mp, R = M*self.w[None], R*self.w[:, None]
            Mbar = np.stack([Mp[:, g].sum(1) for g in self.groups], 1)
        else:
            Mp = M*self.w[None]
            Mbar = np.stack([M[:, g].sum(1) for g in self.groups], 1)
        Rbar = np.stack([R[g].sum(0) for g in self.groups])
        if self.norm_exp == 1:
            Q = R/np.maximum(Rbar+self.k, 1e-300)
        else:                                                        # R_i / Rbar^p, R in units of the median Rbar
            s = np.median(Rbar)
            Q = (R/s)/np.maximum((Rbar+self.k)/s, 1e-300)**self.norm_exp
        return (((Mp[:, :, None]-Mbar[:, :, None]*Q[None]))**2).sum(1)

    def flow(self, M, vmax=4.0, dv=0.1, chunk=64):
        """Local velocity (n_points, 2) = argmin of the cost, refined by a parabola on the grid."""
        g, VX, VY = self.velocity_grid(vmax, dv)
        R = self.predicted(VX.ravel(), VY.ravel())
        n = len(g)
        est = np.zeros((len(M), 2))
        for s in range(0, len(M), chunk):
            cost = self.cost(M[s:s+chunk], R)
            for p, b in enumerate(cost.argmin(1)):
                iy, ix = divmod(b, n)
                c = cost[p].reshape(n, n)
                ox = _parabola(c[iy, ix-1], c[iy, ix], c[iy, ix+1]) if 0 < ix < n-1 else 0
                oy = _parabola(c[iy-1, ix], c[iy, ix], c[iy+1, ix]) if 0 < iy < n-1 else 0
                est[s+p] = g[ix]+ox*dv, g[iy]+oy*dv
        return est

    # ---------------- pooled over points: information-matrix combination ----------------
    def grouped(self, M, R):
        """Reshape to the (.., axis, filter-in-axis) layout used by optic_flow_corrected.info_components."""
        order = np.argsort(self.axis_id, kind='stable')
        A, B = len(self.axes), len(self.mu)//len(self.axes)
        return M[:, order].reshape(len(M), A, B), R[order].T.reshape(R.shape[1], A, B)

    def global_velocity(self, M, vmax=3.0, n_v=41, prior_rel=0.0, **kw):
        """One velocity for all points: local likelihoods combined by their information
        matrices (+ optional slow-motion prior), as in optic_flow_corrected.py."""
        from optic_flow_corrected import info_components, solve_info
        g = np.linspace(-vmax, vmax, n_v)
        VX, VY = np.meshgrid(g, g, indexing='xy')
        Mg, Rg = self.grouped(M*self.w[None], self.predicted(VX.ravel(), VY.ravel())*self.w[:, None])
        SL, SLm = info_components(Mg, Mg.shape[1], Mg.shape[2], Rg, VX, VY, **kw)
        return solve_info(SL, SLm, prior_rel)


def _parabola(a, b, c):
    d = a-2*b+c
    return 0.5*(a-c)/d if d > 0 else 0.0
