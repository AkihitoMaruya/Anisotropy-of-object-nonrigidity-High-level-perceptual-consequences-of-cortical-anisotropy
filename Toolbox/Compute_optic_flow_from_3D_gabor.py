#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:06:22 2024

@author: akihitomaruya
"""


import numpy as np
import os
import matplotlib.pyplot as plt
import numpy.matlib

import matplotlib.colors

from scipy.ndimage.filters import convolve as filter2
import torch.nn.functional as F
global max_abs

import torch

import moviepy.video.io.ImageSequenceClip
from pyr3D_Gabor import pyr3D_Gabor
import os

current_folder = os.path.dirname(os.path.abspath(__file__))

dtype=torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Heeger_pyr_flow(object):
    """ Apply Steerable pyramid in 3D to the given video and compute optic flow. Returns vector flow pattern U and V"""
    
    def __init__(self, UorA='U',sname=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures_paper', 'work', 'Steer_pyr_opticflow'),
    vfile=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures_paper', 'work')+'/',name='Optic_flow_H_pyr',scale=.5,prior=np.ones((211,211)),ex_idx=[]
    ,smoothness=False,alpha=100,num_cells=np.ones(16),fps=10/4,sz_x=200,sz_y=200,Vis=True,full=False,sigma=1,num_scales=1):
        """
       Input: U or A (Uniform or Anisotropic)
        """
        
        self.smoothness=smoothness
        self.alpha=alpha
        self.ex_idx=ex_idx
        path=current_folder+'/Data/'
        if UorA=='U':
            
            self.Ris=np.load(path+'Uniform_Ris.npy')
        if UorA=='U_pyr':
            
            self.Ris=np.load(path+'Uniform_Ris.npy')
        if UorA=='A':
            
           
            self.Ris=np.load(path+'Aniso_Ris.npy')
            
        if UorA=='A_pyr':
            
    
            self.Ris=np.load(path+'Aniso_Ris.npy')
        self.full=full
        self.num_orientations=16
        self.num_ang_intime=3
        self.num_scale=len(self.Ris)//(16*self.num_ang_intime)
        self.pyr=pyr3D_Gabor(UorA,self.num_scale)
        self.sigma=sigma
        self.scale=scale
        self.sname=sname
        self.vfile=vfile
        self.name=name
        self.Vis=Vis
        self.scale=scale
        self.fps=fps
        self.sz_x=sz_x
        self.sz_y=sz_y
        self.num_cells=num_cells
        self.prior=prior

        try:
            os.mkdir(sname)
        except:
            print('File exists')
            
        try:
            os.mkdir(vfile)
        except:
            print('File exists')
    def center_surround(self,kernel_size=33, sigma=1.0):
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        x, y = torch.meshgrid(x, y)
        kernel = torch.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        return (kernel / kernel.max()).unsqueeze(0).unsqueeze(0),(1-kernel / kernel.max()).unsqueeze(0).unsqueeze(0)



    def Compute_Divisive_Norm(self,Res_U_E,sigma=5,k=0.125,sz_surround=11):
        
        central,surround=self.center_surround(sz_surround,sigma=sigma)
        Res_U_E_=Res_U_E.view(Res_U_E.shape[0]*Res_U_E.shape[1],1,Res_U_E.shape[-2],Res_U_E.shape[-1])
        Res_U_S=F.conv2d(Res_U_E_,surround,padding=(sz_surround//2, sz_surround//2))
        
        Res_U_C=F.conv2d(Res_U_E_,central,padding=(sz_surround//2, sz_surround//2))
        Res_U_denom = torch.sqrt(Res_U_C**2 + Res_U_S**2 + k)
        Res_U=(Res_U_C/Res_U_denom).view(Res_U_E.shape)
        return Res_U

    def forward(self,luminance_images):
        self.luminance_images=luminance_images.squeeze()
        self.height=luminance_images.squeeze().shape[1]
        self.width=luminance_images.squeeze().shape[2]
        self.duration=luminance_images.squeeze().shape[0]
        # compute pyr
        pyr_coeffs = self.pyr.forward(luminance_images)
        self.pyr_coeffs =pyr_coeffs 
        # First compute the Ri for each filter
        Ri_all=[]
        u_hat=np.zeros_like(luminance_images.squeeze())
        v_hat=np.zeros_like(luminance_images.squeeze())
        u=np.linspace(-100,100,211)
        v=np.flip(np.linspace(-100,100,211))
        Directions=np.arange(-180,180,360/self.num_orientations)
        Directions=[Directions for ii in range(self.num_ang_intime) for kk in range(self.num_scale)]
        Directions=np.hstack(Directions)
        self.Directions=Directions
        Num_cells=[]
        ME_outputs_all=[]
        ii=0
        for k in pyr_coeffs.keys():
            if isinstance(k, tuple):
                Ri=self.Ris[ii,:,:]
                Ri_all.append(Ri)
                #ME_outputs_all.append(self.Compute_Divisive_Norm(pyr_coeffs[k].abs()**2))
                ME_outputs_all.append(pyr_coeffs[k].abs()**2)
                Num_cells.append(self.num_cells[k[2]])
                ii=ii+1
        self.Num_cells=Num_cells
        # Ri_all.pop(-1)
        # ME_outputs_all.pop(-1)
        self.ME_outputs_all=ME_outputs_all
        self.Ri_all=Ri_all
        #% Make Ri_bar and mi_bar
        Ri_bar_all=[]
        Mi_bar_all=[]
        for ii in range(len(Ri_all)):
            sp_dir=np.abs(Directions[ii])
            idxs=np.argwhere((np.abs(Directions[ii]-Directions)==180) | (np.abs(Directions[ii]-Directions)==0))
            Ri_bar_all.append(np.sum(np.array(Ri_all)[idxs],axis=0).squeeze())
            Mi_bar_all.append(np.sum(np.array(ME_outputs_all)[idxs],axis=0)[0])
        self.Mi_bar_all=Mi_bar_all
        self.Ri_bar_all=Ri_bar_all
        #%% Compute the optic flow 
        if self.full:
            ex_idx=np.argwhere((luminance_images.squeeze()!=np.inf))[:]
            know_idx=False
            prev_tt=0
            num=0
        elif len(self.ex_idx)==0:
            ex_idx=np.argwhere(luminance_images.squeeze()>0.85)[:]
            know_idx=False
            prev_tt=0
            num=0
        elif len(self.ex_idx)>0:
            know_idx=True
            ex_idx=self.ex_idx
            
        
        self.ex_idx=ex_idx
        
        ttt=0
        if know_idx:
            U=np.zeros_like(ex_idx[2:,:]).reshape(self.duration,-1)
            V=np.zeros_like(ex_idx[2:,:]).reshape(self.duration,-1)
            prev_tt=0
            num=0
        for ii in range(ex_idx.shape[1]):
            
            tt=ex_idx[0,ii]
            hh=ex_idx[1,ii]
            ww=ex_idx[2,ii]
            if (know_idx) & (prev_tt<tt):
                num=0
                
           
            inside=np.zeros((211,211))
            for kk in range(len(Ri_all)):
                inside+=(ME_outputs_all[kk][0,tt,hh,ww].numpy()-Mi_bar_all[kk][0,tt,hh,ww]*Ri_all[kk]/Ri_bar_all[kk])**2*Num_cells[kk]
            inside=inside/np.sum(inside)
            
            power=len(Ri_all)
            likelihood=1/((2*np.pi)**(power/2)*self.sigma**power)*np.exp(-1/(2*self.sigma**2)*inside)
            likelihood=likelihood*self.prior/np.sum(likelihood*self.prior)
          
            
            if np.sum(likelihood)!=np.sum(likelihood):
                v_hat[tt,hh,ww]=0
                u_hat[tt,hh,ww]=0
            elif (np.abs(v[np.argwhere(likelihood==np.max(likelihood))[0][0]])>90) |  (np.abs(u[np.argwhere(likelihood==np.max(likelihood))[0][1]])>90):
                v_hat[tt,hh,ww]=0
                u_hat[tt,hh,ww]=0
            else:
                v_hat[tt,hh,ww]=v[np.argwhere(likelihood==np.max(likelihood))[0][0]]
                u_hat[tt,hh,ww]=u[np.argwhere(likelihood==np.max(likelihood))[0][1]]
            prev_tt=tt
            if know_idx:
                U[tt,num]=u_hat[tt,hh,ww]
                V[tt,num]=v_hat[tt,hh,ww]
            num=num+1
        if self.smoothness:
            u = np.zeros((self.duration,self.height,self.width))
            v = np.zeros((self.duration,self.height,self.width))
            alpha = self.alpha
            delta = 10**-2
            for tt in range(self.duration):
                # Initialize u and v
                
                fx=u_hat[tt,:,:]
                fy=v_hat[tt,:,:]
                ft=-(u_hat[tt,:,:]**2+v_hat[tt,:,:]**2)
                # Average kernel
                avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                                            [1 / 6, 0, 1 / 6],
                                            [1 / 12, 1 / 6, 1 / 12]], float)
                #avg_kernel=np.ones((100,100))/(100*100)
                iter_counter = 0
                while True:
                    iter_counter += 1
                    u_avg = filter2(u[tt,:,:], avg_kernel)
                    v_avg = filter2(v[tt,:,:], avg_kernel)
                    p = fx * u_avg + fy * v_avg + ft
                    d = 4 * alpha**2 + fx**2 + fy**2
                    prev = u[tt,:,:]
                
                    u[tt,:,:] = u_avg - fx * (p / d)
                    v[tt,:,:] = v_avg - fy * (p / d)
                
                    diff = np.linalg.norm(u[tt,:,:] - prev, 2)
                    #converges check (at most 300 iterations)
                    if  diff < delta or iter_counter > 1000:
                        # print("iteration number: ", iter_counter)
                        break
                
            u_hat=u
            v_hat=v
            U=u_hat[ex_idx[0,:],ex_idx[1,:],ex_idx[2,:]]
            V=v_hat[ex_idx[0,:],ex_idx[1,:],ex_idx[2,:]]
            
        if self.Vis:
            self.visualize(u_hat, v_hat, ex_idx,self.scale)
        
        return u_hat,v_hat,U,V
    
    def visualize(self,u_hat,v_hat,ex_idx,scale=1):
        def vector_to_rgb(angle, absolute):
            """Get the rgb value for the given `angle` and the `absolute` value
            
            Parameters
            ----------
            angle : float
            The angle in radians
            absolute : float
            The absolute value of the gradient
            
            Returns
            -------
            array_like
            The rgb value as a tuple with values [0..1]
            """
            global max_abs
            # normalize angle
            angle = angle % (2 * np.pi)
            if angle < 0:
                angle += 2 * np.pi
            
            return matplotlib.colors.hsv_to_rgb((angle / 2 / np.pi, 
                                          absolute / max_abs, 
                                          absolute / max_abs))
        global max_abs   
        interval=3
        ttt=0
        for tt in range(self.duration):
            ids=np.argwhere(ex_idx[0,:]==tt)
            if len(ids)==0:
                print('')
            else:
                
                HH=ex_idx[1,ids].reshape(-1)
                WW=ex_idx[2,ids].reshape(-1)
                U_hat=u_hat[tt,HH,WW].reshape(-1)
                V_hat=v_hat[tt,HH,WW].reshape(-1)
                HH=HH[::interval]
                WW=WW[::interval]
                U_hat=U_hat[::interval]
                V_hat=V_hat[::interval]
                
                lengths=np.sqrt(U_hat**2+V_hat**2)
                Mean_lengths=np.mean(lengths)
                U_hat[lengths>Mean_lengths*2]=0
                V_hat[lengths>Mean_lengths*2]=0
                lengths=np.sqrt(U_hat**2+V_hat**2)
                
                
                # fig = plt.figure(frameon=False)
                # fig.set_size_inches(luminance_images[tt,:,:].shape[1]/dpi, luminance_images[tt,:,:].shape[0]/dpi)
                
                px = 1/plt.rcParams['figure.dpi']
                fig=plt.figure(figsize=(self.sz_x*px, self.sz_y*px))
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.imshow(self.luminance_images[ttt,:,:],cmap='gray',vmin=self.luminance_images.min(),vmax=self.luminance_images.max())
                if not np.sum(lengths)==0:
                    max_abs=np.max(lengths)
                    c=np.array(list(map(vector_to_rgb, np.arctan2(V_hat,-U_hat), lengths)))
                    plt.quiver(WW,HH,U_hat,V_hat,color=c,scale=scale, angles='xy',
                    scale_units='xy', linewidths=5,headwidth=20)
                plt.axis('off')
                if ttt<10:
                    sname=self.sname+'/'+self.name+f'_0000{ttt}.jpg'
                elif ttt>99 and tt<1000:
                    sname=self.sname+'/'+self.name+f'_00{ttt}.jpg'
                elif ttt>999:
                    sname=self.sname+'/'+self.name+f'_0{ttt}.jpg'
                elif ttt>9999:
                    sname=self.sname+'/'+self.name+f'_{ttt}.jpg'
                else:
                    sname=self.sname+'/'+self.name+f'_000{ttt}.jpg'
                fig.savefig(sname,facecolor=fig.get_facecolor())
                plt.close('all')    
                ttt=ttt+1
            
        
        video_name=self.vfile+self.name+'.mp4'
        fps=self.fps
        
        image_files2 = [self.sname+'/'+img for img in sorted(os.listdir(self.sname)) if img.endswith(".jpg")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files2, fps=fps)
        clip.write_videofile(video_name) 