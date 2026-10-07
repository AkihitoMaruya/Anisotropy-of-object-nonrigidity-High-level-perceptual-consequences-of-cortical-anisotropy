"""

Compute convolution between 3D gabor filters and the video stimulus


"""


from collections import OrderedDict
import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import os
current_folder = os.path.dirname(os.path.abspath(__file__))
complex_types = [torch.cdouble, torch.cfloat]


class pyr3D_Gabor(nn.Module):
    

    def __init__(self, UorA='U',num_scales=1):

        super().__init__()
        dtype=torch.float32
        self.num_scales=num_scales
        self.num_time=3
        self.num_orientations=16
        self.order=self.num_orientations//2-1
        self.fft_norm = "backward"
        path=current_folder+'/Data/'
        if UorA=='U':
            
           
            self.Gabors=torch.from_numpy(np.load(path+'Uniform_gabs.npy')).type(dtype)
        if UorA=='U_pyr':
            
           
            self.Gabors=torch.from_numpy(np.load(path+'Uniform_pyr.npy')).type(dtype)
        if UorA=='A':
            
           
            self.Gabors=torch.from_numpy(np.load(path+'Aniso_gabs.npy')).type(dtype)
        if UorA=='A_pyr':
            
           
            self.Gabors=torch.from_numpy(np.load(path+'Aniso_pyr.npy')).type(dtype)
    
    

    def forward(self, x):
        
        pyr_coeffs = OrderedDict()
       
        
        imdft = fft.fftn(x, dim=(-3,-2,-1), norm = self.fft_norm)
        imdft = fft.fftshift(imdft)
        
       
        for i in range(self.num_scales):
            for tt in range(self.num_time):
                for b in range(self.num_orientations):
                    mask=self.Gabors[i,tt,b,:,:,:,:]
                    complex_const = np.power(complex(0, -1), self.order)
                
                    banddft = complex_const * mask*imdft 
                    # fft output is then shifted to center frequencies
                    band = fft.ifftshift(banddft)
                    # ifft is applied to recover the filtered representation in spatial domain
                    band = fft.ifftn(band, dim=(-3,-2,-1), norm=self.fft_norm).to(torch.float32)
                    
                    pyr_coeffs[(i, tt,b)] = band
                    
            

        return pyr_coeffs