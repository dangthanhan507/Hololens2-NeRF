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

from tqdm import tqdm

'''
    NOTE: this is a huge dataset.... might be wise to decrease resolution and fps of hololens data if we want to finish this project.
    NOTE: remember to get calibrations to plug in real focal length
    NOTE: create graphs to visualize this stuff

'''

if __name__ == '__main__':
    BATCH_SIZE = 3_000
    EPOCHS = 1
    
    print('Setting up network')
    embedding_xyz = Embedding(3,10).cuda()
    embedding_dir = Embedding(3,4).cuda()
    
    nerf_coarse = NeRF().cuda()
    nerf_fine = NeRF().cuda()
    print('Done setting up Network\n')
    
    print('Loading Dataset')
    dataset = HololensSimpleDataset('../Hololens2-Capture/datanerf/',3, img_wh=(424,240))
    
    dataloader = DataLoader(dataset,shuffle=True,num_workers=4,batch_size=BATCH_SIZE, pin_memory=True)
    print('Done Loading Dataset\n')
    
    
    print('Loading Optimizer/Loss')
    adam = torch.optim.Adam([ {'params': nerf_coarse.parameters()}, {'params': nerf_fine.parameters()}], lr=1e-3)
    loss = nn.MSELoss(reduction='mean')
    print('Done Loading Optimizer/Loss\n')
    
    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]
    
    print('Start Training:\n')
    for epoch in range(EPOCHS):
        print('EPOCH: ', epoch)
        for idx,batch in enumerate( tqdm(dataloader) ):
            adam.zero_grad()
            
            rays = batch['rays'].cuda()
            rgbs = batch['rgbs'].cuda()
            
            #run through rays
            results = defaultdict(list)
            B = rays.shape[0]
            
            #runs NeRF all the way
            rendered_rays = render_rays(models, embeddings, rays,N_importance=64,chunk=B,white_back=False) #simple rendering (more hparams)
            
            for k,v in rendered_rays.items():
                results[k] = torch.cat([v],0)
                
            mse = loss(results['rgb_coarse'],rgbs) + loss(results['rgb_fine'],rgbs) #calc loss
            
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            
            mse.backward()
            adam.step()
    os.makedirs('./models/',exist_ok=True)
    chkpt = {'NeRF Coarse Params': nerf_coarse.state_dict(), 'NeRF Fine Params': nerf_fine.state_dict()}
    torch.save(chkpt, os.path.join('./models','finished.pt'))