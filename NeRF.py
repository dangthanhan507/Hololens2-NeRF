import torch
import torch.nn as nn
import numpy as np

'''
    File
    
    NeRF.py:
    ============

    Part 1 of Project (Implement Neural Radiance Fields):
        (camera poses) -> (RGBsigma)

    Implementation Details:
    ========================
    - Take camera poses and convert pose into (x,viewing direction)

    # Network representation
    - MLP Network is F_theta : (x,d) -> (c,sigma)
        -> F_theta is the MLP network with theta parameters
        -> x is the camera position
        -> d is the viewing direction
        -> c is the color (its prediction depends on both x and d)
        -> sigma is the intensity (its prediction only depends on x)

    # Color representation
    - C(r) is the expected color of camera ray r.
        - C(r) is approximated with stratified sampling and discretizing ray segments
        - basically we estimate Chat(r) by using volumetric rendering which describes C(r)
    
    # Loss
    - Our loss is calculated using exclusively C(r)
        - L = sum[ (C_coarse(r) - C(r))^2 + (C_fine(r) - C(r))^2 ]

        
    Extra Implementations:
    ======================
        -> Hierarchical Sampling
        -> Positional Encoding
'''

class NeRF(nn.Module):
    def __init__(self, xyz_hidden_depth=8, xyz_hidden_dim=256, view_hidden_dim=128):
        super().__init__() #do init with nn.Module

        #Neural Radiance Field Architecture as described in paper
        '''
            Architecture details:
                input: (x,d)

                x is processed with 8 hidden layers (size 256 and ReLU activation) -> (feature,sigma)
                
                store sigma

                concatenate featrue and d -> (feature,d)
                pass (feature,d) into one connected hidden layer (size 128 and ReLU) -> (RGB)

                store RGB
        '''
        layers = []

        xyz_dim = 3
        view_dim = 2
        rgb_dim = 3
        for i in range(xyz_hidden_depth):
            if i == 0:
                #NOTE: some people add an embedding (positional encoding) first
                #NOTE: some people also add in skip connections?
                layers.append(nn.Linear(xyz_dim,xyz_hidden_dim))
            else:
                layers.append(nn.Linear(xyz_hidden_dim,xyz_hidden_dim))
            layers.append(nn.ReLU(True))
        self.xyz_proc_layers = nn.Sequential(*layers)

        self.sigma_proc = nn.Linear(xyz_hidden_dim, 1)

        self.view_proc = nn.Sequential(nn.Linear(xyz_hidden_dim+view_dim, view_hidden_dim),
                                       nn.ReLU(True),
                                       nn.Linear(view_hidden_dim,rgb_dim),
                                       nn.Sigmoid())
        

    def forward(self, x):
        '''
            Argument:
            =========
            x: (Batch Size [optional], 5 [xyz,view])
            
            
            Returns:
            ========
        '''
        xyz = x[:,:3]
        view = x[:,3:]

        feature = self.xyz_proc_layers(xyz)

        sigma = self.sigma_proc(feature)

        feature_view = torch.cat((feature,view), -1)
        rgb = self.view_proc(feature_view)

        rgbsigma = torch.cat((rgb,sigma), -1)
        return rgbsigma
    
if __name__ == '__main__':
    '''
        Unittest Neural Radiance Fields:
    '''

    #test creation of model
    print('Testing creation of NeRF')
    model = NeRF()
    print('\tNo error when creating model\n\n')

    #test syntax error
    print('Testing single input')
    inp = torch.zeros((1,5)) #one vec
    out = model(inp)
    print('\tNo eror when running sample vector on model')
    print('\tOutput Shape:', out.shape)
    assert out.shape == (1,4)
    print('\tShape of output is correct\n\n')

    #test batch of rays
    print('Testing batch input')
    inp = torch.zeros((100,5))
    out = model(inp)
    print('\tNo eror when running sample vector on model')
    print('\tOutput Shape:', out.shape)
    assert out.shape == (100,4)
    print('\tShape of output is correct\n\n')