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
    NOTE: remember to get calibrations to plug in real focal length

'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="")
args = parser.parse_args()


if __name__ == '__main__':
    model_folder = 'models'
    dataset_folder = 'datanerf/'
    
    google_colab = False

    BATCH_SIZE = 1_500
    EPOCHS = 10
    
    print('Setting up network')
    embedding_xyz = Embedding(3,10).cuda()
    embedding_dir = Embedding(3,4).cuda()
    
    nerf_coarse = NeRF().cuda()
    nerf_fine = NeRF().cuda()
    print('Done setting up Network\n')

    if google_colab:
      root = '/content/drive/MyDrive/nerf_desktop/Hololens2-NeRF/'
    else:
      root = './'
    os.makedirs(os.path.join(root,model_folder),exist_ok=True)
    os.makedirs(os.path.join(root,f'{model_folder}_error'),exist_ok=True)

    if args.model != "":
      print('\tLoaded parameters')
      params = torch.load(os.path.join(root,model_folder,args.model), 'cuda')
      nerf_coarse.load_state_dict(params['NeRF Coarse Params'])
      nerf_fine.load_state_dict(params['NeRF Fine Params'])
    
    print('Loading Dataset')
    items = os.listdir(os.path.join(root,model_folder))
    
    path = os.path.join(root,dataset_folder)
    error_path = os.path.join(root,f'{model_folder}_error',f'error{len(items)}.txt')
    
    file_error = open(error_path,'w')
    
    dataset = HololensSimpleDataset(path,3, img_wh=(424,240),image_limit=50,skips=2)
    
    dataloader = DataLoader(dataset,shuffle=True,num_workers=4,batch_size=BATCH_SIZE, pin_memory=True)
    print('Done Loading Dataset\n')
    
    
    print('Loading Optimizer/Loss')
    adam = torch.optim.Adam([ {'params': nerf_coarse.parameters()}, {'params': nerf_fine.parameters()}], lr=1e-3)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(adam, [10,10,10], gamma=0.5)
    loss = nn.MSELoss(reduction='mean')
    print('Done Loading Optimizer/Loss\n')
    
    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]
    
    print('Start Training:\n')
    for epoch in range(EPOCHS):
        print('EPOCH: ', epoch)
        epoch_error = 0
        for idx,batch in enumerate( tqdm(dataloader) ):
            adam.zero_grad()
            
            rays = batch['rays'].cuda()
            rgbs = batch['rgbs'].cuda()
            
            #run through rays
            results = defaultdict(list)
            B = rays.shape[0]
            
            #runs NeRF all the way
            rendered_rays = render_rays(models, embeddings, rays,N_samples=64,N_importance=64,chunk=B,white_back=False) #simple rendering (more hparams)
            
            for k,v in rendered_rays.items():
                results[k] = torch.cat([v],0)
                
            mse = loss(results['rgb_coarse'],rgbs) + loss(results['rgb_fine'],rgbs) #calc loss
            epoch_error += mse
            #mse is a number
            #take this and use for printing
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            
            mse.backward()
            adam.step()
        scheduler.step()
        print(f'Training error: {epoch_error}')
        file_error.write(f'{epoch_error}\n')
        
    file_error.close()
    chkpt = {'NeRF Coarse Params': nerf_coarse.state_dict(), 'NeRF Fine Params': nerf_fine.state_dict()}
    
    torch.save(chkpt, os.path.join(root,model_folder,f'finished{len(items)}.pt'))
