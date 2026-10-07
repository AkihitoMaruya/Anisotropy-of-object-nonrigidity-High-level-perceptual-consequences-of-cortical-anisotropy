#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:13:29 2024

@author: akihitomaruya
"""

import warnings
from collections import OrderedDict
import numpy as np
from scipy.special import factorial

import torch
import torch.fft as fft
import torch.nn as nn

import math

complex_types = [torch.cdouble, torch.cfloat]
def H_filt(r):
    H=torch.zeros_like(r)
    H[r.abs()>=torch.pi/2]=1
    H[r.abs()<=torch.pi/4]=0
    mask=((r.abs() > (torch.pi / 4)) & (r.abs() < (torch.pi / 2)))
    H[mask]=torch.cos(torch.pi / 2 * torch.log2((r[mask].abs() *2) / torch.pi))
    return H
def L_filt(r):
    L=torch.zeros_like(r)
    L[r.abs()<=torch.pi/4]=1
    L[r.abs()>=torch.pi/2]=0
    mask=(r.abs() > (torch.pi / 4)) & (r.abs() < (torch.pi / 2))
    L[mask]=torch.cos(torch.pi / 2 * torch.log2((r[mask].abs() * 4) / torch.pi))
    return L
def Ang_filter(k, K, Alpha, ang_coorxy, two_sided=True):
    # Two-sided: real-symmetric angular filter (lobes at both center and center+pi).
    # One-sided: keeps only the +center lobe; combined with a one-sided radial product
    # this gives an analytic-signal filter whose ifft has even-phase in .real and
    # odd-phase in .imag, so |band| is the Hilbert envelope.
    Gk1 = torch.zeros_like(ang_coorxy)
    center1 = k * np.pi / K
    circular_distance1 = (ang_coorxy - center1 + np.pi) % (2 * np.pi) - np.pi
    mask1 = np.abs(circular_distance1) < np.pi / 2
    Gk1[mask1] = Alpha * torch.cos(circular_distance1[mask1]) ** (K - 1)

    if not two_sided:
        return Gk1

    Gk2 = torch.zeros_like(ang_coorxy)
    center2 = np.pi * (k - K) / K
    circular_distance2 = (ang_coorxy - center2 + np.pi) % (2 * np.pi) - np.pi
    mask2 = np.abs(circular_distance2) < np.pi / 2
    Gk2[mask2] = Alpha * torch.cos(circular_distance2[mask2]) ** (K - 1)

    return Gk1 + Gk2


class Steerable_Pyramid_Freq3D(nn.Module):
    r"""Steerable frequency pyramid in Torch

    Construct a steerable pyramid on matrix 3 dimensional signals, in the
    Fourier domain. Boundary-handling is circular. Reconstruction is exact
    (within floating point errors). However, if the image has an odd-shape,
    the reconstruction will not be exact due to boundary-handling issues
    that have not been resolved.

    The squared radial functions tile the Fourier plane with a raised-cosine
    falloff. Angular functions are cos(theta-k*pi/order+1)^(order).

    Notes
    -----
    
    Parameters
    ----------
    video_shape : `list or tuple`
        shape of input image
    height : 'auto' or `int`
        The height of the pyramid. If 'auto', will automatically determine
        based on the size of `video`.
    order : `int`.
        The Gaussian derivative order used in the spatial domain for the steerable filters, in [1, 15].
        Note that to achieve steerability the minimum number of orientation is `order` + 1,
        and is used here. order +1 indicates that the number of orientation in the spatial domain
        
    order_temp: 'int'
        The Gaussian derivative order used in the temporal domain for the steerable filters, in [1, 15].
        Note that to achieve steerability the minimum number of orientation is `order_temp` + 1,
        and is used here. order_temp +1 indicates that the number of orientation in the temporal domain
    
    is_complex : `bool`
        Whether the pyramid coefficients should be complex or not. If True, the real and imaginary
        parts correspond to a pair of even and odd symmetric filters. If False, the coefficients
        only include the real part / even symmetric filter.
    downsample: `bool`
        Whether to downsample each scale in the pyramid or keep the output pyramid coefficients
        in fixed bands of size imshapeximshape. When downsample is False, the forward method returns a tensor.
    


    """

    def __init__(self, video_shape, height='auto', order=3, order_temp=3, is_complex=False,
                  downsample=True, tight_frame=True,
                  hilbert_axes=('spatial', 'temporal')):
        """
        is_complex : bool, default False
            If True, the pyramid returns complex bands obtained by applying
            POST-HOC Hilbert transforms (on configurable axes) to the real
            bands of a clean real two-sided steerable pyramid. The underlying
            filter geometry is always the same real, tight, K*S-band pyramid
            (regardless of is_complex), and recon_pyr is always exact -- it
            just takes 2*Re(...) of each complex band to recover the real
            band, then runs the standard real-pyramid recon. is_complex=True
            only changes what is RETURNED in pyr_coeffs (and what is_complex
            stats are available), not the recon path.

        tight_frame : bool, default True (deprecated, kept for API compat)
            The real pyramid is always tight; this flag has no effect now.

        hilbert_axes : tuple of str, default ('spatial', 'temporal')
            Only consulted when is_complex=True. Each entry is one of
            {'spatial', 'temporal'}. For each requested axis the pyramid
            applies a post-hoc Hilbert transform on the corresponding
            half-spectrum:
              'spatial'  : half-plane indicator aligned with the band's
                           angular orientation theta_b on the (fx, fy) plane.
                           Stored in pyr_coeffs[(i, b, t)] (= c_S).
              'temporal' : half-axis indicator on f_z > 0 (i.e., f_t > 0).
                           Stored in self._c_T[(i, b, t)] (accessed via the
                           get_temporal_analytic() helper).
            Magnitudes match the legacy is_complex=True, tight_frame=False
            convention (no factor 2 in the half-spectrum indicator), so
            |c_S|^2 = b_ee^2 + b_oe^2 with the same scale as before.

        Filter count is always K spatial * S temporal regardless of mode.

        Magnitude stats:
          |c_S|^2 captures spatial phase invariance: even_S^2 + odd_S^2.
          |c_T|^2 captures temporal phase invariance: even_T^2 + odd_T^2.
          Full 4-component phase-invariant motion magnitude:
            M^2 = b_ee^2 + b_oe^2 + b_eo^2 + b_oo^2
                = |c_S|^2 + |c_T|^2 - Re(c_S)^2 + (Im(c_S) cross terms)
          Use pyr.get_motion_components(i, b, t) for the four sub-bands.

        Reconstruction:
          recon_pyr() takes pyr_coeffs (possibly modified by the caller) and
          extracts the real band as 2*Re(c) if is_complex, else uses the band
          as-is. Then runs the standard real-pyramid recon. Exact recon as
          long as the modification preserves Re = real_band / 2 for the
          complex case.
        """

        super().__init__()

        self.pyr_size = OrderedDict()
        self.order = order
        self.order_temp=order_temp
        self.video_shape = video_shape
        self.tight_frame = bool(tight_frame)

        if (self.video_shape[0] % 2 != 0) or (self.video_shape[1] % 2 != 0) or (self.video_shape[2] % 2 != 0):
            warnings.warn(
                "Reconstruction will not be perfect with odd-sized images")
        max_ht = np.floor(np.log2(min(self.video_shape[0], self.video_shape[1],self.video_shape[2])))-2
        if height == 'auto':
            self.num_scales = int(max_ht)
        elif height > max_ht:
            raise ValueError(
                "Cannot build pyramid higher than %d levels." % (max_ht))
        else:
            self.num_scales = int(height)
        self.is_complex = is_complex
        self.downsample = downsample  
        self.fft_norm = "backward"
        
        if self.order > 15 or self.order <= 0:
            raise ValueError("order must be an integer in the range [1,15].")
        # K = order + 1, S = order_temp + 1. Always K spatial * S temporal
        # filters (real two-sided bilobed). The pyramid is always a tight
        # frame and recon_pyr is always exact. is_complex=True applies post-
        # hoc Hilbert on configurable axes (hilbert_axes); it does NOT change
        # the filter geometry or band count.
        self._K_cos = int(self.order + 1)
        self._S_cos = int(self.order_temp + 1)
        self.num_orientations = self._K_cos
        if self.order_temp > 15 or self.order_temp <= 0:
            raise ValueError("order must be an integer in the range [1,15].")
        self.num_orientations_temp = self._S_cos

        # Hilbert-axes config -- only consulted for is_complex=True.
        valid_axes = {'spatial', 'temporal'}
        hilbert_axes = tuple(hilbert_axes)
        for a in hilbert_axes:
            if a not in valid_axes:
                raise ValueError(f"hilbert_axes entry {a!r} must be one of {valid_axes}.")
        self.hilbert_axes = hilbert_axes
        self._has_spatial_hilbert = 'spatial' in hilbert_axes
        self._has_temporal_hilbert = 'temporal' in hilbert_axes
        
        dims = np.array(self.video_shape)
        
        # Let's create low and high masks
        
        r_z=torch.linspace(-torch.pi, torch.pi, dims[0]+1)[:-1]
        r_y=torch.linspace(-torch.pi, torch.pi, dims[1]+1)[:-1]
        r_x=torch.linspace(-torch.pi, torch.pi, dims[2]+1)[:-1]
        Z,Y,X=torch.meshgrid(r_z,r_y,r_x, indexing='ij')
        
        X=X.unsqueeze(0)#.unsqueeze(0)
        Y=Y.unsqueeze(0)#.unsqueeze(0)
        Z=Z.unsqueeze(0)#.unsqueeze(0)
        R=torch.sqrt(X**2+Y**2+Z**2)/2
        
        lo0mask = L_filt(R)
        hi0mask =H_filt(R)
        
        self.lo0mask = lo0mask
        self.hi0mask = hi0mask
        
        # Normalization term for the angle mask. Same Alpha works in every mode:
        #  - is_complex=False: K two-sided filters tile the full circle with
        #    sum_b |G_b|^2 = alpha^2 cos^{2(K-1)} summed over 2K lobes = 1.
        #  - is_complex=True, tight_frame=True: K one-sided forward filters
        #    (2 * cos^(K-1) on a single pi-wide lobe) paired with K bilobed
        #    two-sided recon filters. Forward*recon = 2*alpha^2*cos^(2(K-1))
        #    on the forward lobe (and 0 on the opposite lobe). Summed over
        #    K filters: sys(theta) + sys(theta+pi) = 2 (pointwise) by the
        #    cosine-tiling identity, because the indicators of the K forward
        #    lobes at theta and at theta+pi together cover the full circle.
        K = self._K_cos
        S = self._S_cos
        Alpha = 2**(K-1) * math.factorial(K-1) / np.sqrt(K * math.factorial(2 * (K-1)))
        AlphaT = 2**(S-1) * math.factorial(S-1) / np.sqrt(S * math.factorial(2 * (S-1)))
        
        # pre-generate the angle, hi and lo masks, as well as the
        # indices used for down-sampling
        self._anglemasks = []
        self._anglemasks_recon = []
        self._himasks = []
        self._lomasks = []
        self._loindices = []
        # Post-hoc Hilbert half-plane indicators (only filled if is_complex).
        # self._half_spatial[i][b]: 0/1 mask of shape matching the scale's grid,
        #   = 1 where cos(theta_b)*fx + sin(theta_b)*fy > 0  (positive-half along
        #   the b-th spatial orientation).
        # self._half_temporal[i]: 0/1 mask of shape matching the scale's grid,
        #   = 1 where f_z > 0  (positive-half along the time axis).
        self._half_spatial = []
        self._half_temporal = []
        
        # need a mock image to down-sample so that we correctly
        # construct the differently-sized masks
        mock_video = torch.rand(*self.video_shape).unsqueeze(0).unsqueeze(0)
        viddft = fft.fftshift(fft.fftn(mock_video, dim=(-3, -2, -1),norm=self.fft_norm))
        lodft = viddft * lo0mask
        
        # this list, used by coarse-to-fine optimization, gives all the
        # scales (including residuals) from coarse to fine
        self.scales = (['residual_lowpass'] + list(range(self.num_scales))[::-1] +
                       ['residual_highpass'])
        self._ldfts=[]
        lomask=lo0mask.clone()
        ldft=lomask
        flta_sys=torch.zeros_like(mock_video).squeeze(0)
        # `lostart`/`loend` from each downsample are relative to the just-exited
        # cropped frame, not the original. Track cumulative offsets so the
        # per-iter flta_sys block and the post-loop residual-lowpass placement
        # land at the correct position in the original-frame coordinate system.
        cum_offset = np.array([0, 0, 0])
        cur_block_size = np.array([hi0mask.shape[-3], hi0mask.shape[-2], hi0mask.shape[-1]])
        for i in range(self.num_scales):
            # First make himask
            R=R*2
            himask=H_filt(R)
            self._himasks.append(himask)
            ang_coorxy=torch.arctan2(Y,X)
            
            anglemasks = []
            anglemasks_recon = []
            half_spatial_per_band = []
            # Temporal half-plane indicator at this scale: f_z > 0.
            # (Same for all bands at this scale.) Use 0.5 at the boundary
            # f_z = 0 so that the conjugate-pair-fixed-point plane gets the
            # correct half-weight and 2*Re(...) recovers the real band
            # exactly (not just up to a missing DC-plane contribution).
            if self.is_complex and self._has_temporal_hilbert:
                half_temp_mask = ((Z > 0).to(torch.float32)
                                  + 0.5 * (Z == 0).to(torch.float32))
                self._half_temporal.append(half_temp_mask)
            else:
                self._half_temporal.append(None)

            # Real two-sided bilobed masks in every mode. The pyramid is
            # always a tight, real, K*S-band steerable pyramid. is_complex
            # only governs what we apply POST-HOC on top of these real
            # bands (see forward()), it does NOT change the mask shape here.
            # spatial orientation
            for b in range(self.num_orientations):
                anglemask_sp = Ang_filter(b, K, Alpha, ang_coorxy, two_sided=True)
                anglemask_recon_sp = anglemask_sp
                center1 = b * np.pi / K
                center2 = np.pi * (b - K) / K
                ang_coorxz = torch.arctan2(Z, np.cos(center1) * X + np.sin(center1) * Y)

                # Per-band spatial half-plane indicator aligned with theta_b:
                # 1 where (cos(theta_b)*fx + sin(theta_b)*fy) > 0, 0 on the
                # negative side, and 0.5 on the boundary plane. The 0.5
                # weight at the boundary is what makes 2*Re(c_S) recover the
                # real band exactly (the boundary plane is the fixed point
                # of (fx, fy) -> (-fx, -fy), so the standard Hermitian-pair
                # conjugate-symmetry argument fails there unless given
                # half-weight).
                if self.is_complex and self._has_spatial_hilbert:
                    proj = np.cos(center1) * X + np.sin(center1) * Y
                    half_sp_mask = ((proj > 0).to(torch.float32)
                                    + 0.5 * (proj == 0).to(torch.float32))
                else:
                    half_sp_mask = None
                half_spatial_per_band.append(half_sp_mask)

                anglemasks_tmp = []
                anglemasks_recon_tmp = []

                # temporal orientation -- two-sided as well
                for t in range(self.num_orientations_temp):
                    anglemask_tmp = Ang_filter(t, S, AlphaT, ang_coorxz, two_sided=True)
                    anglemask_recon_tmp = anglemask_tmp

                    anglemask =anglemask_sp*anglemask_tmp
                    anglemask_recon =anglemask_recon_tmp*anglemask_recon_sp
                    end = cum_offset + cur_block_size
                    flta_sys[:, cum_offset[0]:end[0], cum_offset[1]:end[1], cum_offset[2]:end[2]] += (anglemask*ldft*himask)**2
                    anglemasks_tmp.append(anglemask)
                    anglemasks_recon_tmp.append(anglemask_recon)

                anglemasks.append(anglemasks_tmp)
                anglemasks_recon.append(anglemasks_recon_tmp)

            self._anglemasks.append(anglemasks)
            self._anglemasks_recon.append(anglemasks_recon)
            self._half_spatial.append(half_spatial_per_band)
            if not self.downsample:
                prev_lomask=ldft.clone()
                lomask =L_filt(R)
                ldft=prev_lomask*lomask
                self._lomasks.append(lomask)
                self._loindices.append([np.array([0, 0,0]), dims])
                lodft = lodft * lomask
            else:
                # Make subsample indices
                prev_lomask=ldft.clone()
                dims = np.array([lodft.shape[-3], lodft.shape[-2],lodft.shape[-1]])
                ctr = np.ceil((dims+0.5)/2).astype(int)
                lodims = np.ceil((dims-0.5)/2).astype(int)
                loctr = np.ceil((lodims+0.5)/2).astype(int)
                lostart = ctr - loctr
                loend = lostart + lodims
                self._loindices.append([lostart, loend])

                R=R[:,lostart[0]:loend[0], lostart[1]:loend[1],lostart[2]:loend[2]]
                X=X[:,lostart[0]:loend[0], lostart[1]:loend[1],lostart[2]:loend[2]]
                Y=Y[:,lostart[0]:loend[0], lostart[1]:loend[1],lostart[2]:loend[2]]
                Z=Z[:,lostart[0]:loend[0], lostart[1]:loend[1],lostart[2]:loend[2]]
                lomask =L_filt(R)
                ldft=prev_lomask[:,lostart[0]:loend[0], lostart[1]:loend[1],lostart[2]:loend[2]]*lomask
                self._lomasks.append(lomask)
                # subsampling
                lodft = lodft[:,:,lostart[0]:loend[0], lostart[1]:loend[1],lostart[2]:loend[2]]
                # convolution in spatial domain
                lodft = lodft * lomask

                # Maintain cumulative offset into the original-frame coordinate
                # system, so the next iter's flta_sys block (and the post-loop
                # residual placement) land at the correct position.
                cum_offset = cum_offset + lostart
                cur_block_size = lodims
            self._ldfts.append(ldft)
        ldft_upscaled=torch.zeros_like(mock_video).squeeze(0)
        end = cum_offset + cur_block_size
        ldft_upscaled[:, cum_offset[0]:end[0], cum_offset[1]:end[1], cum_offset[2]:end[2]] = ldft
        
        flta_sys = flta_sys + ldft_upscaled ** 2 + hi0mask ** 2
        # Tight-frame check: pyramid is always real two-sided so should be
        # tight pointwise.
        max_diff = float(torch.max(torch.abs(flta_sys - 1.0)).item())
        if max_diff > 1e-3:
            warnings.warn(
                f"Steerable_Pyramid_Freq3D system-function deviates from 1.0 "
                f"by up to {max_diff:.4f}. Pyramid is not a tight frame; "
                "reconstruction will be approximate."
            )
        self.to(torch.float32)
    def _apply(self, fn, *args, **kwargs):
        r"""Hook that PyTorch's `.to()/.cuda()/.cpu()/.mps()` machinery calls
        on every child module. Because our angular/radial masks are stored as
        plain Python attributes (not nn.Parameter / nn.Buffer), the default
        Module._apply would never touch them — so a parent's `.to('mps')` would
        leave them on CPU and the first forward pass would device-mismatch.

        We override it to apply `fn` (the move/cast function) to every mask
        tensor we hold, in addition to whatever `super()._apply` does for
        registered parameters and buffers.
        """
        super()._apply(fn, *args, **kwargs)
        self.lo0mask = fn(self.lo0mask)
        self.hi0mask = fn(self.hi0mask)
        self._himasks = [fn(m) for m in self._himasks]
        self._lomasks = [fn(m) for m in self._lomasks]
        for i in range(len(self._anglemasks)):
            for j in range(len(self._anglemasks[i])):
                self._anglemasks[i][j] = [fn(m) for m in self._anglemasks[i][j]]
                self._anglemasks_recon[i][j] = [fn(m) for m in self._anglemasks_recon[i][j]]
        if hasattr(self, '_ldfts'):
            self._ldfts = [fn(m) for m in self._ldfts]
        if hasattr(self, '_half_spatial'):
            self._half_spatial = [
                [fn(m) if m is not None else None for m in per_scale]
                for per_scale in self._half_spatial
            ]
        if hasattr(self, '_half_temporal'):
            self._half_temporal = [fn(m) if m is not None else None for m in self._half_temporal]
        return self
    
    def forward(self, x, scales=[]):
        r"""Generate the steerable pyramid coefficients for a video

        Parameters
        ----------
        x : torch.Tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,ch,
            time, height, width).
        scales : list, optional
            Which scales to include in the returned representation. If
            an empty list (the default), we include all
            scales. Otherwise, can contain subset of values present in
            this model's ``scales`` attribute (ints from 0 up to
            ``self.num_scales-1`` and the strs 'residual_highpass' and
            'residual_lowpass'. Can contain a single value or multiple
            values. If it's an int, we include all orientations from
            that scale. Order within the list does not matter.

        Returns
        -------
        representation:
            Pyramid coefficients

        """
        pyr_coeffs = OrderedDict()
        # Side-table for the temporal-Hilbert analytic bands (only filled
        # when is_complex=True and 'temporal' in hilbert_axes). Keyed by
        # (scale, b, t). Caller can access via self.get_temporal_analytic.
        self._c_T = OrderedDict()
        self.x=x
        if not isinstance(scales, list) and not isinstance(scales, tuple):
            raise Exception("scales must be a list!")
        if not scales:
            scales = self.scales
        scale_ints = [s for s in scales if isinstance(s, int)]
        if len(scale_ints) != 0:
            assert (max(scale_ints) < self.num_scales) and (
                min(scale_ints) >= 0), "Scales must be within 0 and num_scales-1"
        
        lo0mask = self.lo0mask.clone()
        hi0mask = self.hi0mask.clone()

        # x is a torch tensor batch of images of size [N,C,W,H]
        assert len(x.shape) == 5, "Input must be batch of images of shape BxCxHxW"
        
        viddft = fft.fftn(x, dim=(-3,-2,-1), norm = self.fft_norm)
        viddft = fft.fftshift(viddft)
        
        if 'residual_highpass' in scales:
            # high-pass
            hi0dft = viddft * hi0mask
            hi0 = fft.ifftshift(hi0dft, dim=(-3, -2, -1))
            hi0 = fft.ifftn(hi0, dim=(-3,-2,-1), norm=self.fft_norm)
            pyr_coeffs['residual_highpass'] = hi0.real
            self.pyr_size['residual_highpass'] = tuple(hi0.real.shape[-3:])
            # hi0dft_recon=viddft * hi0mask**2
        
         #input to the next scale is the low-pass filtered component
        lodft = viddft * lo0mask  
        #ldft=lo0mask.clone()
        # banddft_recon=torch.zeros_like(viddft).to(torch.complex64)
        # lostart=np.array([0,0,0])
        # loend=np.array([hi0mask.shape[-3],hi0mask.shape[-2],hi0mask.shape[-1]])
        for i in range(self.num_scales):

            if i in scales:
                #high-pass mask is selected based on the current scale
                himask = self._himasks[i]
                
                #compute filter output at each orientation
                for b in range(self.num_orientations):
                    for t in range(self.num_orientations_temp):
                    # band pass filtering is done in the fourier space as multiplying by the fft of a gaussian derivative.
                    # The oriented dft is computed as a product of the fft of the low-passed component,
                    # the precomputed anglemask (specifies orientation), and the precomputed hipass mask (creating a bandpass filter)
                    # the complex_const variable comes from the Fourier transform of a gaussian derivative.
                    # Based on the order of the gaussian, this constant changes.
                    
                        
                        anglemask = self._anglemasks[i][b][t]
                        # Real-pyramid band: bilobed two-sided angle mask, no
                        # phase rotation. Result is real-valued (up to FFT
                        # roundoff). This is the "b_ee" base band used by recon.
                        banddft = lodft * anglemask * himask

                        if not self.is_complex:
                            band = fft.ifftshift(banddft, dim=(-3, -2, -1))
                            band = fft.ifftn(band, dim=(-3,-2,-1), norm=self.fft_norm)
                            pyr_coeffs[(i, b,t)] = band.real
                            self.pyr_size[(i, b,t)] = tuple(band.shape[-3:])
                        else:
                            # Post-hoc Hilbert(s) on configurable axes. Each
                            # Hilbert is implemented as a half-spectrum truncation
                            # of the band's FFT (no factor 2 -> matches the legacy
                            # is_complex=True magnitude convention). The recon
                            # path will undo the half-spectrum factor with 2*Re.
                            if self._has_spatial_hilbert:
                                half_sp = self._half_spatial[i][b]
                                c_S_dft = banddft * half_sp
                                c_S = fft.ifftshift(c_S_dft, dim=(-3, -2, -1))
                                c_S = fft.ifftn(c_S, dim=(-3,-2,-1), norm=self.fft_norm)
                                pyr_coeffs[(i, b,t)] = c_S
                                self.pyr_size[(i, b,t)] = tuple(c_S.shape[-3:])
                            else:
                                # No spatial Hilbert requested -- store real band
                                band = fft.ifftshift(banddft, dim=(-3, -2, -1))
                                band = fft.ifftn(band, dim=(-3,-2,-1), norm=self.fft_norm)
                                pyr_coeffs[(i, b,t)] = band.real
                                self.pyr_size[(i, b,t)] = tuple(band.shape[-3:])

                            if self._has_temporal_hilbert:
                                half_t = self._half_temporal[i]
                                c_T_dft = banddft * half_t
                                c_T = fft.ifftshift(c_T_dft, dim=(-3, -2, -1))
                                c_T = fft.ifftn(c_T, dim=(-3,-2,-1), norm=self.fft_norm)
                                self._c_T[(i, b, t)] = c_T
                if not self.downsample:
                    # no subsampling of angle and rad
                    # just use lo0mask
                    lomask = self._lomasks[i]
                    lodft = lodft * lomask
                    
                    # because we don't subsample here, if we are not using orthonormalization that
                    # we need to manually account for the subsampling, so that energy in each band remains the same
                    # the energy is cut by factor of 4 so we need to scale magnitudes by factor of 2
                    
                    # if self.fft_norm != "ortho":
                    #     lodft = 2*lodft
                else:
                    
                    # subsample indices
                    lostart, loend = self._loindices[i]
                        
                    # subsampling of the dft for next scale
                    lodft = lodft[:, :, lostart[0]:loend[0], lostart[1]:loend[1],lostart[2]:loend[2]]
                    # low-pass filter mask is selected
                    lomask = self._lomasks[i]
                    # again multiply dft by subsampled mask (convolution in spatial domain)
    
                    lodft = lodft * lomask
                #ldft=self._ldfts[i]
                    
                
        if 'residual_lowpass' in scales:
            # compute residual lowpass when height <=1
            # lo0_dft_recon=torch.zeros_like(hi0mask.unsqueeze(0)).to(torch.complex64)
            # lo0_dft_recon[:, :, lostart[0]:loend[0], lostart[1]:loend[1],lostart[2]:loend[2]]=ldft*lodft
            lo0 = fft.ifftshift(lodft, dim=(-3, -2, -1))
            lo0 = fft.ifftn(lo0, dim=(-3,-2,-1), norm=self.fft_norm)
            pyr_coeffs['residual_lowpass'] = lo0.real
            self.pyr_size['residual_lowpass'] = tuple(lo0.real.shape[-3:])
        # recon_dft=hi0dft_recon+banddft_recon+lo0_dft_recon
        # recon=fft.ifftn(fft.ifftshift(recon_dft),dim=(-3,-2,-1),norm=self.fft_norm).real
        return pyr_coeffs
                
    def recon_pyr(self, pyr_coeffs, levels='all', bands='all',bands_tmp='all', levels_noise=[], bands_noise=[],bands_tmp_noise=[], noise_level=0.1, threshold_ratio=0.1):
        
        """Reconstruct the video or batch of video, optionally using subset of pyramid coefficients.

        NOTE: in order to call this function, you need to have
        previously called `self.forward(x)`, where `x` is the tensor you
        wish to reconstruct. This will fail if you called `forward()`
        with a subset of scales.

        Parameters
        ----------
        pyr_coeffs : `OrderedDict`
            pyramid coefficients to reconstruct from
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_lowpass'`. If `'all'`, returned value will contain all
            valid levels. Otherwise, must be one of the valid levels.
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations. 
        bands_tmp : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations_temp-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations. 
        levels_noise : `list`, `int`,  or {`'all'`, `'residual_highpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_lowpass'`. If `'all'`, returned value will contain all
            valid levels. Otherwise, must be one of the valid levels.
        bands_noise : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations. 
        bands_tmp_noise : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations_temp-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations. 
        noise_level: float (the level of noise)
        threshold_ratio: 0-1 (specifies the region where the noise is introduced)
        
        

        Returns
        -------
        recon : `torch.Tensor`
            The reconstructed image or batch of images.
            Output is of size BxCxHxW
    
        """
        # For reconstruction to work, last time we called forward needed
        # to include all levels
        for s in self.scales:
            if isinstance(s, str):
                if s not in pyr_coeffs.keys():
                    raise Exception(f"scale {s} not in pyr_coeffs! pyr_coeffs must include"
                                    " all scales, so make sure forward() was called with arg "
                                    "scales=[]")
            else:
                for b in range(self.num_orientations):
                    for t in range(self.num_orientations_temp):
                        if (s, b,t) not in pyr_coeffs.keys():
                            raise Exception(f"scale {s} not in pyr_coeffs! pyr_coeffs must "
                                            "include all scales, so make sure forward() was called "
                                            "with arg scales=[]")

       

        recon_keys = self._recon_keys(levels, bands,bands_tmp)
        recon_keys_noise=self._recon_keys(levels_noise, bands_noise,bands_tmp_noise)
        self.recon_keys_noise=recon_keys_noise
        self.recon_keys=recon_keys
        #print(recon_keys_noise)
        # load masks from model
        lo0mask = self.lo0mask
        hi0mask = self.hi0mask
        recon_dft=torch.zeros_like(self.x).to(torch.complex64)
        if 'residual_highpass' in recon_keys:
            high_coeffs=pyr_coeffs['residual_highpass']
            if 'residual_highpass' in recon_keys_noise:
                high_coeffs=self.add_noise_to_filter_response_3d(high_coeffs,noise_level,threshold_ratio)
                
            high_coeffs_dft=fft.fftshift(fft.fftn(high_coeffs,dim=(-3,-2,-1),norm=self.fft_norm))
            hi0dft_recon=high_coeffs_dft*hi0mask
            recon_dft=hi0dft_recon
        banddft_recon=torch.zeros_like(self.x).to(torch.complex64)
        lostart=np.array([0,0,0])
        loend=np.array([hi0mask.shape[-3],hi0mask.shape[-2],hi0mask.shape[-1]])
        ldft=lo0mask
        ctr=np.array(self.video_shape)//2
        
        
        for i in range(self.num_scales):
            #high-pass mask is selected based on the current scale
            himask = self._himasks[i]
            
            #compute filter output at each orientation
            for b in range(self.num_orientations):
                for t in range(self.num_orientations_temp):
                    if (i, b,t) in recon_keys:
                        # Mask is the same bilobed-real two-sided mask used in
                        # forward (no asymmetric forward/recon now).
                        anglemask = self._anglemasks_recon[i][b][t]
                        lw=anglemask.squeeze().shape
                        band_coeffs=pyr_coeffs[(i, b,t)]
                        if (i, b,t) in recon_keys_noise:
                            band_coeffs=self.add_noise_to_filter_response_3d(band_coeffs,noise_level,threshold_ratio)

                        # Extract the underlying real band.
                        # For is_complex=True with spatial Hilbert: stored
                        # c_S = ifft(banddft_real * half_spatial), with
                        # banddft_real = X * G_real * H * L. The half-spectrum
                        # truncation halves the amplitude, so Re(c_S) =
                        # real_band / 2, and 2*Re(c_S) = real_band recovers
                        # the underlying real-pyramid band exactly.
                        # For is_complex=False: band_coeffs is already
                        # real_band (real-valued tensor) -- pass through.
                        if self.is_complex and self._has_spatial_hilbert:
                            real_band = 2.0 * band_coeffs.real
                        else:
                            real_band = band_coeffs.real if band_coeffs.is_complex() else band_coeffs

                        banddft=fft.fftshift(fft.fftn(real_band,dim=(-3,-2,-1),norm=self.fft_norm))

                        banddft_recon[:, :,
              ctr[0] - lw[0] // 2: ctr[0] + (lw[0] + 1) // 2,
              ctr[1] - lw[1] // 2: ctr[1] + (lw[1] + 1) // 2,
              ctr[2] - lw[2] // 2: ctr[2] + (lw[2] + 1) // 2] += anglemask*himask*ldft*banddft
            lostart, loend = self._loindices[i]
            ldft=self._ldfts[i]
        recon_dft=recon_dft+banddft_recon
        lo0_dft_recon=torch.zeros_like(self.x).to(torch.complex64)
        if 'residual_lowpass' in recon_keys:
            
            lo_coeffs=pyr_coeffs['residual_lowpass']
            if 'residual_lowpass' in recon_keys_noise:
                lo_coeffs=self.add_noise_to_filter_response_3d(lo_coeffs,noise_level,threshold_ratio)
            lodft=fft.fftshift(fft.fftn(lo_coeffs,dim=(-3,-2,-1),norm=self.fft_norm))
            lw=ldft.squeeze().shape
            lo0_dft_recon[:, :, 
              ctr[0] - lw[0] // 2: ctr[0] + (lw[0] + 1) // 2,
              ctr[1] - lw[1] // 2: ctr[1] + (lw[1] + 1) // 2,
              ctr[2] - lw[2] // 2: ctr[2] + (lw[2] + 1) // 2]=ldft*lodft
            recon_dft=recon_dft+lo0_dft_recon
        
        recon=fft.ifftn(fft.ifftshift(recon_dft),dim=(-3,-2,-1),norm=self.fft_norm).real
        return recon
    
    # def add_noise_to_filter_response_3d(self, filter_response, noise_level=0.1, threshold_ratio=0.1):
    #     """
    #     Add multiplicative noise around the region where the filter responds in the Fourier domain for 3D data.
    
    #     Parameters:
    #     filter_response: torch.Tensor
    #         The 3D filter response in the Fourier domain (shape: [D, H, W]).
    #     noise_level: float
    #         The scale of the noise to be added.
    #     threshold_ratio: float
    #         Ratio of the maximum response magnitude to determine the response region.
    
    #     Returns:
    #     noisy_response: torch.Tensor
    #         The 3D filter response with added multiplicative noise.
    #     """
    #     # Compute the magnitude of the filter response
    #     magnitude = torch.abs(filter_response)
    
    #     # Threshold to define the region of response
    #     threshold = threshold_ratio * torch.max(magnitude)
    
    #     # Create a mask for the region of response
    #     response_region = magnitude > threshold
    
    #     # Generate random multiplicative noise within the region
    #     noise = torch.normal(mean=1.0, std=noise_level, size=filter_response.shape, device=filter_response.device)
    
    #     # Apply the mask to the noise to confine it to the response region
    #     noise = 1 + (noise - 1) * response_region
    
    #     # Ensure the noise has a mean of 1 in the response region
    #     if response_region.sum() > 0:
    #         mean_noise = noise[response_region].mean()
    #         noise = noise * (1 / mean_noise)
    
    #     # Apply the multiplicative noise to the filter response
    #     noisy_response = filter_response * noise
    
    #     return noisy_response
    def add_noise_to_filter_response_3d(self,filter_response, noise_level=0.1, threshold_ratio=0.1):
          """
          Add zero-mean noise around the region where the filter responds in the Fourier domain for 3D data.
     
          Parameters:
          filter_response: torch.Tensor
              The 3D filter response in the Fourier domain (shape: [D, H, W]).
          noise_level: float
              The scale of the noise to be added.
          threshold_ratio: float
              Ratio of the maximum response magnitude to determine the response region.
     
          Returns:
          noisy_response: torch.Tensor
              The 3D filter response with added noise.
          """
          # Compute the magnitude of the filter response
          magnitude = torch.abs(filter_response)
     
          # Threshold to define the region of response
          threshold = threshold_ratio * torch.max(magnitude)
     
          # Create a mask for the region of response
          response_region = magnitude > threshold
     
          # Generate random noise within the region
          noise = torch.normal(mean=0.0, std=noise_level*torch.max(magnitude), size=filter_response.shape, device=filter_response.device)
     
          # Apply the mask to the noise to confine it to the response region
          noise = noise * response_region
     
          # Ensure the noise has zero mean in the response region
          if response_region.sum() > 0:
              mean_noise = noise[response_region].mean()
              noise -= mean_noise
     
          # Add the noise to the filter response
          noisy_response = filter_response + noise
     
          return noisy_response
    # def add_noise_to_filter_response_3d(self, filter_response,  threshold_ratio=0.1):
    #     """
    #     Scramble the response around the region where the filter responds in the Fourier domain for 3D data.
    
    #     Parameters:
    #     filter_response: torch.Tensor
    #         The 3D filter response in the Fourier domain (shape: [D, H, W]).
    #     noise_level: float
    #         The scale of the noise to be added (currently unused in scrambling).
    #     threshold_ratio: float
    #         Ratio of the maximum response magnitude to determine the response region.
    
    #     Returns:
    #     scrambled_response: torch.Tensor
    #         The 3D filter response with scrambled values in the response region.
    #     """
    #     # Compute the magnitude of the filter response
    #     magnitude = torch.abs(filter_response)
    
    #     # Threshold to define the region of response
    #     threshold = threshold_ratio * torch.max(magnitude)
    
    #     # Create a mask for the region of response
    #     response_region = magnitude > threshold
    
    #     # Extract the values in the response region
    #     region_values = filter_response[response_region]
    
    #     # Shuffle (scramble) the values within the response region
    #     scrambled_values = region_values[torch.randperm(region_values.numel(), device=filter_response.device)]
    
    #     # Create a copy of the filter response to modify
    #     scrambled_response = filter_response.clone()
    
    #     # Replace the response region with scrambled values
    #     scrambled_response[response_region] = scrambled_values
    
    #     return scrambled_response

    def _recon_levels_check(self, levels):
        r"""Check whether levels arg is valid for reconstruction and return valid version

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        which levels to include. This makes sure those levels are valid and gets them in the form
        we expect for the rest of the reconstruction. If the user passes `'all'`, this constructs
        the appropriate list (based on the values of `pyr_coeffs`).

        Parameters
        ----------
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`, or `'residual_lowpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_highpass'` and `'residual_lowpass'` (if appropriate for the
            pyramid). If `'all'`, returned value will contain all valid levels. Otherwise, must be
            one of the valid levels.

        Returns
        -------
        levels : `list`
            List containing the valid levels for reconstruction.

        """
        if isinstance(levels, str) and levels == 'all':
            levels = ['residual_highpass'] + \
                list(range(self.num_scales)) + ['residual_lowpass']
        else:
            if not hasattr(levels, '__iter__') or isinstance(levels, str):
                # then it's a single int or string
                levels = [levels]
            levs_nums = np.array(
                [int(i) for i in levels if isinstance(i, int) or i.isdigit()])
            assert (levs_nums >= 0).all(
            ), "Level numbers must be non-negative."
            assert (levs_nums < self.num_scales).all(
            ), "Level numbers must be in the range [0, %d]" % (self.num_scales-1)
            levs_tmp = list(np.sort(levs_nums))  # we want smallest first
            if 'residual_highpass' in levels:
                levs_tmp = ['residual_highpass'] + levs_tmp
            if 'residual_lowpass' in levels:
                levs_tmp = levs_tmp + ['residual_lowpass']
            levels = levs_tmp
        # not all pyramids have residual highpass / lowpass, but it's easier to construct the list
        # including them, then remove them if necessary.
        if 'residual_lowpass' not in self.pyr_size.keys() and 'residual_lowpass' in levels:
            levels.pop(-1)
        if 'residual_highpass' not in self.pyr_size.keys() and 'residual_highpass' in levels:
            levels.pop(0)
        return levels

    def _recon_bands_check(self, bands):
        """Check whether bands arg is valid for reconstruction and return valid version

        When reconstructing the input video (i.e., when calling `recon_pyr()`), the user specifies
        which orientations to include. This makes sure those orientations are valid and gets them
        in the form we expect for the rest of the reconstruction. If the user passes `'all'`, this
        constructs the appropriate list (based on the values of `pyr_coeffs`).

        Parameters
        ----------
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to
            `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations.
            Otherwise, must be one of the valid orientations.

        Returns
        -------
        bands: `list`
            List containing the valid orientations for reconstruction.
        """
        if isinstance(bands, str) and bands == "all":
            bands = np.arange(self.num_orientations)
        else:
            bands = np.array(bands, ndmin=1)
            assert (bands >= 0).all(
            ), "Error: band numbers must be larger than 0."
            assert (bands < self.num_orientations).all(
            ), "Error: band numbers must be in the range [0, %d]" % (self.num_orientations - 1)
        return bands
    def _recon_bands_tmp_check(self, bands_tmp):
        """Check whether bands arg is valid for reconstruction and return valid version

        When reconstructing the input video (i.e., when calling `recon_pyr()`), the user specifies
        which orientations in temporal domain to include. This makes sure those orientations are valid and gets them
        in the form we expect for the rest of the reconstruction. If the user passes `'all'`, this
        constructs the appropriate list (based on the values of `pyr_coeffs`).

        Parameters
        ----------
        bands_tmp : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to
            `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations.
            Otherwise, must be one of the valid orientations.

        Returns
        -------
        bands_tmp: `list`
            List containing the valid orientations for reconstruction.
        """
        if isinstance(bands_tmp, str) and bands_tmp == "all":
            bands_tmp = np.arange(self.num_orientations_temp)
        else:
            bands_tmp = np.array(bands_tmp, ndmin=1)
            assert (bands_tmp >= 0).all(
            ), "Error: band numbers must be larger than 0."
            assert (bands_tmp < self.num_orientations_temp).all(
            ), "Error: band numbers must be in the range [0, %d]" % (self.num_orientations_temp - 1)
        return bands_tmp

    def _recon_keys(self, levels, bands,bands_tmp, max_orientations=None,max_orientations_temp=None):
        """Make a list of all the relevant keys from `pyr_coeffs` to use in pyramid reconstruction

        When reconstructing the input video (i.e., when calling `recon_pyr()`), the user specifies
        some subset of the pyramid coefficients to include in the reconstruction. This function
        takes in those specifications, checks that they're valid, and returns a list of tuples
        that are keys into the `pyr_coeffs` dictionary.

        Parameters
        ----------
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`, `'residual_lowpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_highpass'` and `'residual_lowpass'` (if appropriate for the
            pyramid). If `'all'`, returned value will contain all valid levels. Otherwise, must be
            one of the valid levels.
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations.
        max_orientations: `None` or `int`.
            The maximum number of orientations we allow in the reconstruction. when we determine
            which ints are allowed for bands, we ignore all those greater than max_orientations.
        max_orientations: `None` or `int`.
            The maximum number of orientations in temporal domain we allow in the reconstruction. when we determine
            which ints are allowed for bands, we ignore all those greater than max_orientations.
        
        
        Returns
        -------
        recon_keys : `list`
            List of `tuples`, all of which are keys in `pyr_coeffs`. These are the coefficients to
            include in the reconstruction of the image.

        """
        levels = self._recon_levels_check(levels)
        bands = self._recon_bands_check(bands)
        bands_tmp=self._recon_bands_tmp_check(bands_tmp)
        if max_orientations is not None:
            for i in bands:
                if i >= max_orientations:
                    warnings.warn(("You wanted band %d in the reconstruction but max_orientation"
                                    " is %d, so we're ignoring that band" % (i, max_orientations)))
            bands = [i for i in bands if i < max_orientations]
        if max_orientations_temp is not None:
            for i in bands_tmp:
                if i >= max_orientations_temp:
                    warnings.warn(("You wanted band %d in the reconstruction but max_orientation"
                                    " is %d, so we're ignoring that band" % (i, max_orientations_temp)))
            bands_tmp = [i for i in bands_tmp if i < max_orientations_temp]
        
        recon_keys = []
        for level in levels:
            # residual highpass and lowpass
            if isinstance(level, str):
                recon_keys.append(level)
            # else we have to get each of the (specified) bands at
            # that level
            else:
                recon_keys.extend([(level, band,band_tmp) for band in bands for band_tmp in bands_tmp])
        #print(recon_keys)
        return recon_keys

    # ------------------------------------------------------------------
    # Steerability
    # ------------------------------------------------------------------

    @staticmethod
    def _build_steering_matrix(K):
        r"""Build the pyrtools-style steering matrix for K orientations of
        cos^(K-1) angular filters at theta_b = b*pi/K, b=0..K-1.

        Returns
        -------
        mtx : torch.Tensor, shape (K, n_basis)
            Multiply by `steervec` to get the per-band weights for steering.
        harmonics : np.ndarray, length n_h
            Harmonic indices used; n_basis = 2*n_h - (1 if 0 in harmonics else 0).

        Mirrors `pyrtools.steerable_filters` (the 2D version used by your
        snippet `pt.pyramids.steer(level, theta, harmonics, mtx)`).
        """
        K = int(K)
        if K % 2 == 0:
            harmonics = np.arange(1, K + 1, 2)          # 1, 3, ..., K-1
        else:
            harmonics = np.arange(0, K, 2)              # 0, 2, ..., K-1
        angles = np.arange(K, dtype=np.float64) * (np.pi / K)
        cos_rows = np.stack([np.cos(h * angles) for h in harmonics], axis=0)
        sin_rows = np.stack([np.sin(h * angles) for h in harmonics], axis=0)
        if K % 2 == 0:
            imtx = np.concatenate([cos_rows, sin_rows], axis=0)
        else:
            # odd K: the h=0 row of sin is identically zero, skip it
            imtx = np.concatenate([cos_rows, sin_rows[1:]], axis=0)
        mtx = np.linalg.pinv(imtx).astype(np.float32)
        return torch.from_numpy(mtx), harmonics

    def _steering_weights(self, K, theta):
        r"""K-vector of band weights for steering K basis bands to angle `theta`
        (radians). For an even-K cos^(K-1) basis this gives exact steerability
        (the bands fully span the relevant trig-polynomial subspace).
        """
        mtx, harmonics = self._build_steering_matrix(K)
        h = torch.from_numpy(harmonics.astype(np.float32))
        args = h * float(theta)
        if int(K) % 2 == 0:
            steervec = torch.cat([torch.cos(args), torch.sin(args)])
        else:
            steervec = torch.cat([torch.cos(args), torch.sin(args[1:])])
        return mtx @ steervec    # (K,)

    def get_temporal_analytic(self, scale, b, t):
        r"""Return the temporal-Hilbert analytic band c_T = b_ee + i*b_eo for
        the (scale, b, t) cell, computed by the last forward() call.

        Available only when is_complex=True and 'temporal' is in hilbert_axes.
        Re(c_T) = b_ee / 2 = real_band / 2 (matches the magnitude convention
        used by pyr_coeffs's c_S). Im(c_T) is the temporal Hilbert of b_ee.

        See get_motion_components() for a friendlier 4-tuple decomposition.
        """
        if not (self.is_complex and self._has_temporal_hilbert):
            raise RuntimeError(
                "Temporal Hilbert is not active. Pass hilbert_axes=('spatial','temporal') "
                "(or just ('temporal',)) at construction to enable it."
            )
        if not hasattr(self, "_c_T") or (scale, b, t) not in self._c_T:
            raise RuntimeError(
                f"No stored c_T for (scale={scale}, b={b}, t={t}). Call forward() first."
            )
        return self._c_T[(scale, b, t)]

    def get_motion_components(self, pyr_coeffs, scale, b, t):
        r"""Return the four phase-component bands (b_ee, b_oe, b_eo, b_oo) for
        the (scale, b, t) cell, ready for the full phase-invariant motion-
        energy magnitude

            M^2 = b_ee^2 + b_oe^2 + b_eo^2 + b_oo^2

        Requires is_complex=True with both 'spatial' and 'temporal' Hilberts.
        Returns four real-valued tensors with the same shape as the band.

        Notes
        -----
        b_ee = Re(c_S) = Re(c_T)             (the real-pyramid band / 2)
        b_oe = Im(c_S)                       (spatial Hilbert of b_ee)
        b_eo = Im(c_T)                       (temporal Hilbert of b_ee)
        b_oo = temporal Hilbert of b_oe      (= spatial Hilbert of b_eo,
                                              recomputed here on the fly)
        All four are at the magnitude scale where b_ee = real_band / 2 (the
        legacy is_complex=True convention).
        """
        if not (self.is_complex and self._has_spatial_hilbert and self._has_temporal_hilbert):
            raise RuntimeError(
                "Need both 'spatial' and 'temporal' in hilbert_axes for motion components."
            )
        c_S = pyr_coeffs[(scale, b, t)]
        c_T = self._c_T[(scale, b, t)]
        b_ee = c_S.real
        b_oe = c_S.imag
        b_eo = c_T.imag

        # Compute b_oo = temporal Hilbert of b_oe. Implemented as the half-
        # spectrum truncation H_z: ifft(2 * 1[fz>0] * fft(b_oe)).imag.
        # We use the per-scale precomputed half_temporal mask.
        half_t = self._half_temporal[scale]
        boe_dft = fft.fftshift(fft.fftn(b_oe.to(torch.complex64), dim=(-3,-2,-1), norm=self.fft_norm))
        # Multiply by 2*1[fz>0] and ifft; imag part is the Hilbert.
        ana_dft = 2.0 * half_t * boe_dft
        ana = fft.ifftn(fft.ifftshift(ana_dft, dim=(-3,-2,-1)),
                        dim=(-3,-2,-1), norm=self.fft_norm)
        b_oo = ana.imag
        return b_ee, b_oe, b_eo, b_oo

    def motion_energy_sq(self, pyr_coeffs, scale, b, t):
        r"""Full phase-invariant motion-energy magnitude squared for the
        (scale, b, t) cell:
            M^2(z, y, x) = b_ee^2 + b_oe^2 + b_eo^2 + b_oo^2
        Requires is_complex=True with both Hilbert axes active.
        """
        b_ee, b_oe, b_eo, b_oo = self.get_motion_components(pyr_coeffs, scale, b, t)
        return b_ee**2 + b_oe**2 + b_eo**2 + b_oo**2

    def steer_spatial(self, pyr_coeffs, scale, theta_xy, t_index, K_use=None):
        r"""Return the band at spatial orientation `theta_xy` (radians) for
        a fixed temporal-orientation index `t_index`, at the given pyramid scale.

        Parameters
        ----------
        pyr_coeffs : dict
            The dict returned by `forward(vid)`. Bands are keyed as
            (scale, b_spatial, b_temporal).
        scale : int
            Pyramid scale (0 = finest oriented scale).
        theta_xy : float
            Target spatial-azimuth angle in radians.
        t_index : int
            Index of the temporal orientation to hold fixed (0..S-1).
        K_use : int, optional
            How many spatial bands to use. Defaults to self.order+1 (= K).
            The pyramid always stores K spatial bands; this argument is
            kept for API compatibility.
        """
        K_cos = int(self.order + 1)
        K = K_use if K_use is not None else K_cos
        weights = self._steering_weights(K, theta_xy).to(self.lo0mask.device)
        result = None
        for b in range(K):
            band = pyr_coeffs[(scale, b, t_index)]
            term = weights[b] * band
            result = term if result is None else result + term
        return result

    def steer_temporal(self, pyr_coeffs, scale, theta_xz, b_index, S_use=None):
        r"""Return the band at temporal-elevation `theta_xz` (radians) for a
        fixed spatial-orientation index `b_index`, at the given pyramid scale.

        Same pattern as :meth:`steer_spatial` but along the temporal axis.
        """
        S_cos = int(self.order_temp + 1)
        S = S_use if S_use is not None else S_cos
        weights = self._steering_weights(S, theta_xz).to(self.lo0mask.device)
        result = None
        for t in range(S):
            band = pyr_coeffs[(scale, b_index, t)]
            term = weights[t] * band
            result = term if result is None else result + term
        return result

    def steer(self, pyr_coeffs, scale, theta_xy, theta_xz, K_use=None, S_use=None):
        r"""Return the band at arbitrary (theta_xy, theta_xz) by separable
        steering across both spatial and temporal orientations.

        3D analog of `pt.pyramids.steer(level, theta, ...)` -- but with an
        extra angle: steers in BOTH the azimuth and the elevation.

        Parameters
        ----------
        pyr_coeffs : dict
            The dict returned by `forward(vid)`.
        scale : int
            Pyramid scale.
        theta_xy : float
            Spatial-azimuth target (radians).
        theta_xz : float
            Temporal-elevation target (radians). For a fixed spatial direction
            this selects the velocity (0 = stationary, pi/2 = pure flicker).
        K_use, S_use : int, optional
            How many basis filters to use along each axis (default: K, S).

        Notes
        -----
        Pyramid stores K spatial * S temporal bands (K = order+1,
        S = order_temp+1). For is_complex=True the spatial bands are
        one-sided / analytic; for is_complex=False they are real.
        """
        K_cos = int(self.order + 1)
        S_cos = int(self.order_temp + 1)
        K = K_use if K_use is not None else K_cos
        S = S_use if S_use is not None else S_cos
        cb = self._steering_weights(K, theta_xy).to(self.lo0mask.device)
        ct = self._steering_weights(S, theta_xz).to(self.lo0mask.device)
        result = None
        for b in range(K):
            for t in range(S):
                w = cb[b] * ct[t]
                band = pyr_coeffs[(scale, b, t)]
                term = w * band
                result = term if result is None else result + term
        return result


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        