#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:57:48 2023

@author: akihitomaruya
"""
import matplotlib.colors
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import moviepy.video.io.ImageSequenceClip
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



def vis_vec(pos1,pos2,vec1,vec2,name,scale=1,idx=[], path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures_paper', 'work')+'/',
            vfile=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures_paper', 'work')+'/',lim=5):
    global max_abs   
    sz_x=200
    sz_y=200
    duration=np.shape(pos1)[0]
    # First show the optimal vec field 
   
    sfile=path+name
   
    try:
        os.mkdir(sfile)
    except:
        print('File exists')
        
    try:
        os.mkdir(vfile)
    except:
        print('File exists')
    fps=30
    if len(idx)>0:
        prev_tt=np.nan
        for aa in idx:
            tt=aa[0]
            if prev_tt!=tt:
                idx_t=idx[:,0]==0
                px = 1/plt.rcParams['figure.dpi']
                fig=plt.figure(figsize=(sz_x*px, sz_y*px))
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.plot(pos1[tt,:],pos2[tt,:],'ko')
                plt.xlim([-lim,lim])
                plt.ylim([-lim,lim])
                plt.axis('off')
                angles=np.arctan2(-vec2[tt,idx[idx[:,0]==tt,1]],-vec1[tt,idx[idx[:,0]==tt,1]])
                lengths=np.sqrt(vec1[tt,idx[idx[:,0]==tt,1]]**2+vec2[tt,idx[idx[:,0]==tt,1]]**2)
                max_abs=np.max(lengths)
                c=np.array(list(map(vector_to_rgb, angles, lengths)))
                plt.quiver(pos1[tt,idx[idx[:,0]==tt,1]],pos2[tt,idx[idx[:,0]==tt,1]],vec1[tt,idx[idx[:,0]==tt,1]],vec2[tt,idx[idx[:,0]==tt,1]],color=c,scale=scale, angles='xy',
                scale_units='xy', linewidths=5,headwidth=20)
                
                plt.axis('off')
                if tt<10:
                    sname=sfile+'/'+name+f'_0000{tt}.jpg'
                elif tt>99 and tt<1000:
                    sname=sfile+'/'+name+f'_00{tt}.jpg'
                elif tt>999:
                    sname=sfile+'/'+name+f'_0{tt}.jpg'
                elif tt>9999:
                    sname=sfile+'/'+name+f'_{tt}.jpg'
                else:
                    sname=sfile+'/'+name+f'_000{tt}.jpg'
                fig.savefig(sname,facecolor=fig.get_facecolor())
                plt.close('all')    
                prev_tt=tt
            else:
                prev_tt=tt
    else:
        for tt in range(duration):
            px = 1/plt.rcParams['figure.dpi']
            fig=plt.figure(figsize=(sz_x*px, sz_y*px))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.plot(pos1[tt,:],pos2[tt,:],'ko')
            plt.xlim([-1.5,1.5])
            plt.ylim([-1.5,1.5])
            plt.axis('off')
            angles=np.arctan2(-vec2[tt,:],-vec1[tt,:])
            lengths=np.sqrt(vec1[tt,:]**2+vec2[tt,:]**2)
            max_abs=200#np.max(lengths)
            c=np.array(list(map(vector_to_rgb, angles, lengths)))
            plt.quiver(pos1[tt,:],pos2[tt,:],vec1[tt,:],vec2[tt,:],color=c,scale=scale, angles='xy',
            scale_units='xy', linewidths=5,headwidth=20)
            
            plt.axis('off')
            if tt<10:
                sname=sfile+'/'+name+f'_0000{tt}.jpg'
            elif tt>99 and tt<1000:
                sname=sfile+'/'+name+f'_00{tt}.jpg'
            elif tt>999:
                sname=sfile+'/'+name+f'_0{tt}.jpg'
            elif tt>9999:
                sname=sfile+'/'+name+f'_{tt}.jpg'
            else:
                sname=sfile+'/'+name+f'_000{tt}.jpg'
            fig.savefig(sname,facecolor=fig.get_facecolor())
            plt.close('all')    
    video_name=vfile+name+'.mp4'
    fps=fps
    
    image_files2 = [sfile+'/'+img for img in sorted(os.listdir(sfile)) if img.endswith(".jpg")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files2, fps=fps)
    clip.write_videofile(video_name) 

def make_rotating_stim( pos1,pos2,vec1,vec2,sz_x=300,sz_y=300,name='Horizontal_rt',rot=0,scale=1,Vis=0):
    global max_abs   
    
   
    duration=pos1.shape[0]
    sfile=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures_paper', 'work')+'/'+name
    vfile=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures_paper', 'work')+'/'

   

    u=np.linspace(-1.2,1.2,sz_x)
    v=np.linspace(-1.2,1.2,sz_y)
    U_=np.zeros((duration,sz_x,sz_y))
    V_=np.zeros((duration,sz_x,sz_y))
    Video=np.zeros((duration,sz_x,sz_y))
    
    interval=1
    

    try:
        os.mkdir(sfile)
    except:
        print('File exists')
    try:
        os.mkdir(vfile)
    except:
        print('File exists')
    

    ex_idx=[]
    for tt in range(duration):
        P1_=pos1[tt,:]*np.cos(rot*np.pi/180)-pos2[tt,:]*np.sin(rot*np.pi/180)
        P2_=pos1[tt,:]*np.sin(rot*np.pi/180)+pos2[tt,:]*np.cos(rot*np.pi/180)
        U_hat_=vec1[tt,:]*np.cos(rot*np.pi/180)-vec2[tt,:]*np.sin(rot*np.pi/180)
        V_hat_=vec1[tt,:]*np.sin(rot*np.pi/180)+vec2[tt,:]*np.cos(rot*np.pi/180)
        p1=P1_
        p2=P2_
        idx1=np.array([np.argmin((u-p1[ii])**2) for ii in range(len(p1))])
        idx2=np.array([np.argmin((v-p2[ii])**2) for ii in range(len(p2))])
        idx=np.array([np.ones_like(idx1)*tt,sz_y-idx2,idx1])
        ex_idx.append(idx)
        U_[tt,idx2,idx1]=U_hat_
        V_[tt,idx2,idx1]=V_hat_
        Video[tt,idx2,idx1]=1
        Video[tt,:,:]=cv2.GaussianBlur(Video[tt,:,:], (17,17), 0)
        Video[tt,:,:][Video[tt,:,:]>0]=1
        Video[tt,:,:]=cv2.GaussianBlur(Video[tt,:,:], (17,17), 0)
        Video[tt,:,:]=np.flip(Video[tt,:,:],axis=1)
        
        if Vis==1:
        
            lengths=np.sqrt(U_hat_[::interval]**2+V_hat_[::interval]**2)
            px = 1/plt.rcParams['figure.dpi']
            fig=plt.figure(figsize=(sz_x*px, sz_y*px))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            max_abs=np.max(lengths)
            c=np.array(list(map(vector_to_rgb, np.arctan2(V_hat_[::interval],-U_hat_[::interval]), lengths)))
            plt.quiver(idx1[::interval],sz_y-idx2[::interval],U_hat_[::interval],V_hat_[::interval],color=c,scale=scale, angles='xy',
            scale_units='xy', linewidths=100,headwidth=20)
            #plt.plot(P1_[::interval],P2_[::interval],'ko',linewidth=1,markersize=1)
            plt.imshow(Video[tt,:,:],cmap='gray',vmax=Video.max(),vmin=Video.min())
            plt.axis('off')
            # plt.xlim([-1.2,1.2])
            # plt.ylim([-1.2,1.2])
            
            if tt<10:
                sname=sfile+'/'+name+f'_0000{tt}.jpg'
            elif tt>99 and tt<1000:
                sname=sfile+'/'+name+f'_00{tt}.jpg'
            elif tt>999:
                sname=sfile+'/'+name+f'_0{tt}.jpg'
            elif tt>9999:
                sname=sfile+'/'+name+f'_{tt}.jpg'
            else:
                sname=sfile+'/'+name+f'_000{tt}.jpg'
            fig.savefig(sname,facecolor=fig.get_facecolor())
            plt.close('all')    
    if Vis==1:
        import moviepy.video.io.ImageSequenceClip
        video_name=vfile+name+'.mp4'
        image_files2 = [sfile+'/'+img for img in sorted(os.listdir(sfile)) if img.endswith(".jpg")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files2, fps=15)
        clip.write_videofile(video_name) 
        ex_idx=np.hstack(ex_idx)
        
    return Video,ex_idx