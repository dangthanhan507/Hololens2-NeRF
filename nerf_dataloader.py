#all PyTorch stuff
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset

#libraries for parsing data
import cv2
import json
import os
import numpy as np

#setup NeRF rays
from ray_utils import *

'''
    HololensSimpleDataset Notes:
    =============================
        -> we load in nerf dataset folder
        
'''
class HololensSimpleDataset(Dataset):
    def __init__(self, root_dir, focal, split='train', spheric_poses=False, img_wh=(1280,720), bounds=(2,6), image_limit=50):
        '''
            Despite it being just an __init__. we need to parse the dataset here in order to call __len__ and __getitem__
        '''
        self.focal = focal
        self.root_dir = root_dir
        
        self.img_wh = img_wh
        
        self.split = split
        self.white_back = False
        
        self.spheric_poses = spheric_poses
        
        self.transform = T.ToTensor()
        
        near,far = bounds
        self.image_limit = image_limit
        self.setup_data(near,far)
        pass
    def setup_data(self, near, far):
        self.image_paths = sorted(os.listdir(os.path.join(self.root_dir,'pv')))
        
        W,H = self.img_wh
        
        self.directions = get_ray_directions(H,W,self.focal)
        
        with open(os.path.join(self.root_dir,'pv_pose.json')) as f:
            j_pv = json.load(f)
        if self.split == 'train':
            self.all_rays = []
            self.all_rgbs = []
            
            for i, image_path in enumerate(self.image_paths[:self.image_limit]):
                timestamp = image_path.split('.')[0]
                image_path = os.path.join(self.root_dir,'pv',image_path)
                image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB) #(h,w,3)
                c2w = torch.FloatTensor(np.array(j_pv[timestamp]).T)[:3,:] #3x4 matrix for pose
                
                image = self.transform(image) # (3,h,w)
                image = image.view(3,-1).permute(1,0) #(h*w,3)
                
                
                rays_o, rays_d = get_rays(self.directions, c2w)
                #for non spheric_poses use ndc rays
                if not self.spheric_poses:
                    near,far = 0,1
                    rays_o, rays_d = get_ndc_rays(H,W,self.focal, 1.0, rays_o, rays_d)
                else:
                    raise NotImplementedError
                
                self.all_rgbs += [image]
                self.all_rays += [torch.cat([rays_o,rays_d, near*torch.ones_like(rays_o[:,:1]), far*torch.ones_like(rays_o[:,:1])], 1)] #(h*w,8)
                
            self.all_rays = torch.cat(self.all_rays,0) #((N_images)*h*2,8)
            self.all_rgbs = torch.cat(self.all_rgbs,0) #((N_images)*h*2,3)
            
        elif self.split == 'test':
            self.all_rgbs = []
            self.all_rays = []
            for i, image_path in enumerate(self.image_paths[:self.image_limit]):
                timestamp = image_path.split('.')[0]
                image_path = os.path.join(self.root_dir,'pv',image_path)
                image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB) #(h,w,3)
                c2w = torch.FloatTensor(np.array(j_pv[timestamp]).T)[:3,:] #3x4 matrix for pose
                
                image = self.transform(image) # (3,h,w)
                image = image.view(3,-1).permute(1,0) # (h*w,3)
                
                
                rays_o, rays_d = get_rays(self.directions, c2w)
                #for non spheric_poses use ndc rays
                if not self.spheric_poses:
                    near,far = 0,1
                    rays_o, rays_d = get_ndc_rays(H,W,self.focal, 1.0, rays_o, rays_d)
                else:
                    raise NotImplementedError
                
                self.all_rgbs.append(image)
                self.all_rays.append(torch.cat([rays_o,rays_d, near*torch.ones_like(rays_o[:,:1]), far*torch.ones_like(rays_o[:,:1])], 1))
    
    '''
        Required functions to define to inherit from Dataset.
        Overload these functions!
    '''
    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'test':
            return len(self.all_rays)
        return 21
    
    def __getitem__(self, idx):
        sample = None
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        elif self.split == 'test':
            sample = {'rays': self.all_rays[idx], 'rgbs': self.all_rgbs[idx]}
        return sample
    
    
#NOTE: this is where we will add in the tracker and all of that.
class HololensPanopticDataset(Dataset):
    pass