'''
    I took the nerf_pl code for creating a neural radiance field.
    Not worth re-implementing this network if I plan on writing Panoptic Neural Fields.
    
    Dataset: Checked
    NeRF Code: Cloned
    Training: Not yet
'''


# models
from nerf import Embedding, NeRF
from rendering import render_rays
from metrics import *

from torch.utils.data import DataLoader
from nerf_dataloader import HololensSimpleDataset

import os, sys
from collections import defaultdict

import torch
import torch.nn as nn
#define loss
# class MSELoss(nn.Module):
#     def __init__(self):
#         super(MSELoss, self).__init__()
#         self.loss = nn.MSELoss(reduction='mean')

#     def forward(self, inputs, targets):
#         loss = self.loss(inputs['rgb_coarse'], targets)
#         if 'rgb_fine' in inputs:
#             loss += self.loss(inputs['rgb_fine'], targets)

#         return loss

if __name__ == '__main__':
    BATCH_SIZE = 20
    EPOCHS = 1
    embedding_xyz = Embedding(3,10)
    # loss = MSELoss()
    
    loss = nn.MSELoss(reduction='mean')
    
    embedding_xyz = Embedding(3,10)
    embedding_dir = Embedding(3,4)
    
    nerf_coarse = NeRF()
    nerf_fine = NeRF()
    
    dataset = HololensSimpleDataset('../Hololens2-Capture/datanerf/',50)
    
    dataloader = DataLoader(dataset,shuffle=True,num_workers=4,batch_size=BATCH_SIZE, pin_memory=True)
    
    adam = torch.optim.Adam([ {'params': nerf_coarse.parameters()}, {'params': nerf_fine.parameters()}], lr=1e-3)
    
    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]
    for epoch in range(EPOCHS):
        print('\tEPOCH: ', epoch)
        for idx,batch in enumerate(dataloader):
            
            adam.zero_grad()
            
            rays = batch['rays']
            rgbs = batch['rgbs']
            
            #run through rays
            results = defaultdict(list)
            B = rays.shape[0]
            
            #runs NeRF all the way
            rendered_rays = render_rays(models, embeddings, rays,N_importance=64) #simple rendering (more hparams)
            
            for k,v in rendered_rays.items():
                results[k] = torch.cat([v],0)
                
            mse = loss(results['rgb_coarse'],rgbs) + loss(results['rgb_fine'],rgbs) #calc loss
            
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            
            mse.backward()
            adam.step()