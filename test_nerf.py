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

import matplotlib.pyplot as plt

#visualize dpeth
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

@torch.no_grad()
def run_nerf(rays, models, chunks):
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunks):
        rendered_ray_chunks = render_rays(models,embeddings,rays[i:i+chunks],N_importance=64,chunk=chunks,white_back=False, test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]
    
    for k,v in results.items():
        results[k] = torch.cat(v,0)
    return results

if __name__ == '__main__':
    print('Setting up network')
    embedding_xyz = Embedding(3,10).cuda()
    embedding_dir = Embedding(3,4).cuda()

    nerf_coarse = NeRF().cuda()
    nerf_fine = NeRF().cuda()
    print('Done setting up Network\n')

    print('Loading Dataset')
    dataset = HololensSimpleDataset('../Hololens2-Capture/datanerf/',3,img_wh=(424,240),split='test')

    print('Done Loading Dataset\n')

    #load NERF
    with open('./models/finished1.pt', 'rb') as f:
        params = torch.load('./models/finished1.pt', 'cuda')
    nerf_coarse.load_state_dict(params['NeRF Coarse Params'])
    nerf_fine.load_state_dict(params['NeRF Fine Params'])
    
    sample = dataset[0]
    rays = sample['rays'].cuda()
    # rays = rays.reshape(rays.shape).cuda()
    
    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]
    
    print('Running NeRF')
    chunks = 1024*2
    results = run_nerf(rays,models,chunks)
    print('Done Running NeRF')
    
    print('Running Visualize')
    W,H = dataset.img_wh
    im_gt = sample['rgbs'].reshape(H,W,3).cpu().numpy()
    im_pred = results['rgb_fine'].view(H,W,3).cpu().numpy()
    alpha_pred = results['opacity_fine'].view(H,W).cpu().numpy()
    depth_pred = results['depth_fine'].view(H,W)
    print('Visualizing....')
    
    
    plt.subplots(figsize=(15, 8))
    plt.tight_layout()
    plt.subplot(221)
    plt.title('GT')
    plt.imshow(im_gt)
    plt.subplot(222)
    plt.title('pred')
    plt.imshow(im_pred)
    plt.subplot(223)
    plt.title('depth')
    plt.imshow(visualize_depth(depth_pred).permute(1,2,0))
    plt.show()