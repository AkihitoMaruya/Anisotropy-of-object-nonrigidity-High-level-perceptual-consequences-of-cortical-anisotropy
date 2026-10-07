#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: akihitomaruya
"""
import moviepy.video.io.ImageSequenceClip
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import os
from datetime import date

today = date.today()
class TwoRingsStim(object):
    
    def __init__(self,Type='Circ',Motion_type='Rot',im_rot=0,im_size=378,duration=190,fps=60,distance=100,lw=8):
        """
        This code defines a class TwoRingsStim that generates and saves a sequence of images representing a visual stimulus of two rotating rings,
        that either wobble or rotate, and then compiles these images into a video. 
        
        Arguments:
            Type: Can be 'Circ' (circular) or 'Oct' (octagonal).
            Motion_type: Can be 'Rot' (rotation) or 'Wob' (wobble).
            im_rot: The degree of image rotation applied.
            im_size: Size of the generated images.
            duration: The total number of frames in the video.
            fps: Frames per second for the video.
            distance (dc): Distance parameter for calculating 3D perspective.
            lw: Line width of the rings in the images.
        
        
        Returns
        Sequence of images and the corresponding video

        """
        self.Type=Type
        self.Motion_type=Motion_type
        self.im_rot=im_rot
        self.im_sz=im_size
        self.duration=duration
        self.fps=fps
        self.dc=distance
        self.fc=distance
        self.phi=30*np.pi/180
        
        if Type=='Circ':
            self.theta=np.arange(0,np.pi*2,.01)
        elif Type=='Oct':
            self.theta=np.linspace(0,np.pi*2,8,endpoint=False)
        self.Omega=-np.linspace(0,2*np.pi,duration,endpoint=False)
        current_folder = os.getcwd()
        self.sfile = os.path.join(current_folder, 'Images')
        self.vfile = os.path.join(current_folder, 'Videos')
        self.name=Type+f'_rot_{im_rot}_'+self.Motion_type
        self.lw=lw # Line width
        try:
            os.mkdir(self.sfile+'/'+self.name)
        except:
            print('File exists')
        
        try:
            os.mkdir(self.vfile+'/')
        except:
            print('Vfile exists')
    def draw_im(self):
        for tt in range(len(self.Omega)):
            OmegaT=self.Omega[tt]
            
            # Rotation
            if self.Motion_type=='Rot':
                #  the bottom ring
                X1=np.cos(OmegaT)*np.cos(self.theta)-np.sin(OmegaT)*np.sin(self.theta)*np.cos(self.phi)
                Y1=np.sin(self.theta)*np.sin(self.phi)-np.sin(self.phi)
                Z1=-np.sin(OmegaT)*np.cos(self.theta)-np.cos(OmegaT)*np.sin(self.theta)*np.cos(self.phi)
                
                
                # top ring
                X2=np.cos(OmegaT)*np.cos(self.theta)-np.sin(OmegaT)*np.sin(self.theta)*np.cos(-self.phi)
                Y2=np.sin(self.theta)*np.sin(-self.phi)+np.sin(self.phi)
                Z2=-np.sin(OmegaT)*np.cos(self.theta)-np.cos(OmegaT)*np.sin(self.theta)*np.cos(-self.phi)
            if self.Motion_type=='Wob':
                OmegaT=OmegaT+np.pi/2
                X1=(np.cos(OmegaT)**2*np.cos(self.phi)+np.sin(OmegaT)**2)*np.cos(self.theta)-(np.sin(OmegaT)*np.cos(OmegaT)*np.cos(self.phi)-np.sin(OmegaT)*np.cos(OmegaT))*np.sin(self.theta)
                Y1=np.cos(OmegaT)*np.sin(self.phi)*np.cos(self.theta)-np.sin(OmegaT)*np.sin(self.phi)*np.sin(self.theta)-np.sin(self.phi)
                Z1=(np.cos(OmegaT)*np.sin(OmegaT)-np.sin(OmegaT)*np.cos(OmegaT)*np.cos(self.phi))*np.cos(self.theta)+(np.sin(OmegaT)**2*np.cos(self.phi)+np.cos(OmegaT)**2)*np.sin(self.theta)
                
                X2=(np.cos(OmegaT)**2*np.cos(-self.phi)+np.sin(OmegaT)**2)*np.cos(self.theta)-(np.sin(OmegaT)*np.cos(OmegaT)*np.cos(-self.phi)-np.sin(OmegaT)*np.cos(OmegaT))*np.sin(self.theta)
                Y2=np.cos(OmegaT)*np.sin(-self.phi)*np.cos(self.theta)-np.sin(OmegaT)*np.sin(-self.phi)*np.sin(self.theta)+np.sin(self.phi)
                Z2=(np.cos(OmegaT)*np.sin(OmegaT)-np.sin(OmegaT)*np.cos(OmegaT)*np.cos(-self.phi))*np.cos(self.theta)+(np.sin(OmegaT)**2*np.cos(-self.phi)+np.cos(OmegaT)**2)*np.sin(self.theta)
                
            
            u1=X1*self.fc/(self.dc-Z1)
            v1=Y1*self.fc/(self.dc-Z1)
            u2=X2*self.fc/(self.dc-Z2)
            v2=Y2*self.fc/(self.dc-Z2)
            
            
            im_rot=self.im_rot*np.pi/180
            u1r=u1*np.cos(im_rot)-v1*np.sin(im_rot)
            v1r=u1*np.sin(im_rot)+v1*np.cos(im_rot)
            u2r=u2*np.cos(im_rot)-v2*np.sin(im_rot)
            v2r=u2*np.sin(im_rot)+v2*np.cos(im_rot)
            
            
            px = 1/plt.rcParams['figure.dpi']
            fig=plt.figure(figsize=(self.im_sz*px, self.im_sz*px),frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.plot(np.hstack((u1r,u1r[0])),np.hstack((v1r,v1r[0])),color='k',linewidth=self.lw)
            plt.plot(np.hstack((u2r,u2r[0])),np.hstack((v2r,v2r[0])),color='k',linewidth=self.lw)
            plt.xlim([-1.1,1.1])
            plt.ylim([-1.1,1.1])
            plt.axis('off')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.draw()
            if tt<10:
                sname=self.sfile+'/'+self.name+'/'+f'im_0000{tt}.jpg'
            elif tt>99 and tt<1000:
                sname=self.sfile+'/'+self.name+'/'+f'im_00{tt}.jpg'
            elif tt>999:
                sname=self.sfile+'/'+self.name+'/'+f'im_0{tt}.jpg'
            elif tt>9999:
                sname=self.sfile+'/'+self.name+'/'+f'im_{tt}.jpg'
            else:
                sname=self.sfile+'/'+self.name+'/'+f'im_000{tt}.jpg'
            fig.savefig(sname,facecolor=fig.get_facecolor())
            plt.close('all')  
            
    def Make_video(self):
        video_name=self.vfile+'/'+self.name+'.mp4'
        
        image_files2 = [self.sfile+'/'+self.name+'/'+img for img in sorted(os.listdir(self.sfile+'/'+self.name)) if img.endswith(".jpg")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files2, fps=self.fps)
        clip.write_videofile(video_name) 
        
    def forward(self):
        self.draw_im()
        self.Make_video()
        
        
        

            
            
            
            
            
            
            
            
            
            
            