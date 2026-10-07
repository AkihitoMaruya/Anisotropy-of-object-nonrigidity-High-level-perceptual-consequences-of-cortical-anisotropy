# Heeger motion model with Gaussian filters — closed form

Implementation: `heeger_gaussian_flow.py` (this folder). Companion to
`heeger_motion_model.md` / `optic_flow_corrected.py`, which do the same for the steerable
pyramid's cos^(K−1) filters by numerical great-circle integration. With Gaussian filters every
step is analytic.

Notation: frequency $\mathbf f=(f_x,f_y,f_t)$ in cycles/sample, $f_y$ pointing up on the
screen; velocity $\mathbf v=(v_x,v_y)$ in px/frame.

## 1. Filters

Each filter is the product of three Gaussians, in the 3D frequency radius, the spatial angle and
the temporal angle:

$$\widehat H(\mathbf f)=e^{-\frac{(\rho-f_0)^2}{2\sigma_r^2}}\;
e^{-\frac{(\theta-\theta_k)^2}{2\sigma_{\theta,k}^2}}\;
e^{-\frac{(a-\phi_j)^2}{2\sigma_t^2}},\qquad
\rho=\lVert\mathbf f\rVert,\ \theta=\operatorname{atan2}(f_y,f_x),\
a=\operatorname{atan2}\!\big(f_t,\ \cos\theta_k f_x+\sin\theta_k f_y\big).$$

$\sigma_{\theta,k}$ may differ between filters (e.g. cortical orientation anisotropy).

**Linearisation.** At the peak
$\boldsymbol\mu=f_0(\cos\phi_j\cos\theta_k,\ \cos\phi_j\sin\theta_k,\ \sin\phi_j)$ the three
coordinates have gradients $\nabla\rho=\mathbf e_r$,
$\nabla\theta=\mathbf e_\theta/(f_0\cos\phi_j)$, $\nabla a=\mathbf e_a/f_0$ with the orthonormal
frame

$$\mathbf e_r=\boldsymbol\mu/f_0,\quad
\mathbf e_\theta=(-\sin\theta_k,\cos\theta_k,0),\quad
\mathbf e_a=(-\sin\phi_j\cos\theta_k,\,-\sin\phi_j\sin\theta_k,\,\cos\phi_j).$$

Replacing each coordinate by its first-order expansion turns the product into one 3D Gaussian

$$\widehat H(\mathbf f)\approx e^{-\frac12(\mathbf f-\boldsymbol\mu)^\top C^{-1}(\mathbf f-\boldsymbol\mu)},\qquad
C=E\,\mathrm{diag}\!\big(\sigma_r^2,\ (f_0\cos\phi_j\,\sigma_{\theta,k})^2,\ (f_0\sigma_t)^2\big)E^\top,\
E=[\mathbf e_r\ \mathbf e_\theta\ \mathbf e_a].$$

The filter is cut to the half-space $\cos\theta_k f_x+\sin\theta_k f_y>0$, so it is one-sided:
its real and imaginary parts are an even/odd spatial quadrature pair and $|c|^2$ is motion energy.
The Gaussian mass past the cut is
$\Phi\!\big(-f_0\cos\phi_j/\sqrt{\mathbf n^\top C\mathbf n}\big)$, $\mathbf n=(\cos\theta_k,\sin\theta_k,0)$
(< 1% for the banks used), so the closed forms below treat the filter as a full Gaussian.

## 2. Expected motion energy for moving white noise

White noise translating at $\mathbf v$ has power $S_0$ on the motion plane
$f_t=-\mathbf v\cdot(f_x,f_y)$ and zero elsewhere. By Parseval (Heeger Eq. 7)

$$R(\mathbf v)=S_0\iint \big|\widehat H\big(f_x,f_y,-\mathbf v\cdot(f_x,f_y)\big)\big|^2\,df_x\,df_y .$$

$|\widehat H|^2$ is a Gaussian with covariance $S=C/2$. Write the plane normal as
$\mathbf m=(v_x,v_y,1)$, $\mathbf n=\mathbf m/\lVert\mathbf m\rVert$; the area element on the plane
is $dA=\lVert\mathbf m\rVert\,df_x\,df_y$. The integral of an unnormalised Gaussian over a plane
through the origin is the 1D marginal along its normal, evaluated at 0:

$$\boxed{\,R(\mathbf v)=\frac{S_0\,(2\pi)^{3/2}\sqrt{\det S}}{\lVert\mathbf m\rVert}\;
\frac{1}{\sqrt{2\pi\,\mathbf n^\top S\mathbf n}}\;
\exp\!\Big(-\frac{(\mathbf n\cdot\boldsymbol\mu)^2}{2\,\mathbf n^\top S\mathbf n}\Big)\,}$$

The exponent is Heeger's Eq. 8 generalised to any covariance: it is small when the motion
plane passes through the filter's centre ($\mathbf n\cdot\boldsymbol\mu=0$, i.e. $\mu_t=-\mathbf v\cdot\boldsymbol\mu_{sp}$).
In Heeger's bank all filters share one axis-aligned covariance, so the prefactor is the same
for every filter and cancels in the normalisation below; with orientation-dependent widths it
differs between filters and is kept.

## 3. Combining the filters

Group the filters by spatial axis (θ and θ+π, all temporal channels; Heeger's columns),
$\bar m_i=\sum_{j\in M_i}m_j$, $\bar R_i=\sum_{j\in M_i}R_j$ (Heeger Eq. 10). The contrast
$S_0$ cancels, and the least-squares velocity minimises (Eq. 11)

$$\chi^2(\mathbf v)=\sum_i \Big[n_i m_i-\bar m_i\,\frac{n_i R_i(\mathbf v)}{\bar R_i(\mathbf v)}\Big]^2 ,\qquad
\bar m_i=\sum_{j\in M_i}n_j m_j,\ \bar R_i=\sum_{j\in M_i}n_j R_j ,$$

where $n_i$ is the number of cells carrying filter $i$ ($n_i=1$ in Heeger's model), so
$n_i m_i$ and $n_i R_i$ are population energies. Because $n$ is the same for all filters on one
spatial axis, this equals weighting each squared residual of the $n=1$ model by $n_i^2$.

**Measured vs predicted filters.** By default (`filters='product'`) the measured energies use
the exact product of the three Gaussians; the linearised Gaussian is used only for the closed-form
white-noise predictions $R_i(\mathbf v)$. `filters='gaussian'` uses the linearised Gaussian for both.

**Normalisation constant** (`k`): the predicted split can be computed as $R_i/(\bar R_i+k)$
instead of $R_i/\bar R_i$ ($k=0$ is Heeger's model). For $k>0$ the predicted proportions no
longer sum to 1 and the cell numbers no longer cancel within a group.

**Normalisation exponent** (`norm_exp`): $R_i/\bar R_i^{\,p}$, with $R$ in units of the median
$\bar R$ over the velocity grid; $p=1$ is Heeger's model, $p=0$ no normalisation.

**Raw normalisation** (`normalize='raw'`): the measured population energy is compared with a
prediction normalised by the raw filter responses,
$\chi^2=\sum_i\big[n_i m_i-\bar m_i^{\rm raw}R_i/\bar R_i\big]^2$ with $\bar m_i^{\rm raw}=\sum_{j\in M_i}m_j$.
The decoder does not know the cell numbers, so they no longer cancel and can bias the estimate.

- **Pooling** (`energies(..., pool=[(σ_t, σ_s), ...])`): each energy map is blurred with a Gaussian
  window before sampling — Heeger's approximation of the Parseval integral over a local window.
- **Local flow:** minimise $\chi^2$ at each point (`GaussianBank.flow`).
- **Global motion:** turn each point's $\exp(-\chi^2/\tau)$ into a Gaussian with information
  matrix $\Lambda_n$ and combine $\hat{\mathbf v}=(\sum_n\Lambda_n+\lambda I)^{-1}\sum_n\Lambda_n\boldsymbol\mu_n$,
  as in `heeger_motion_model.md` §5–6 (`GaussianBank.global_velocity`, which calls
  `optic_flow_corrected.info_components` / `solve_info`).

## 4. Limits

The predictions are exact for a flat (white-noise) spectrum. A 1D contour puts its power on a
line, not a plane: the within-axis ratios across temporal channels then differ from the
white-noise prediction, which biases the recovered speed along the normal (≈ ×2 at 1 px/frame
for the default bank) while leaving the direction unbiased for an isotropic bank.
