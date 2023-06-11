'''
    File

    utils_ray.py
    =========
    Purpose:
        -> create all the helper functions for working with rays
        -> create all the helper functions for working with color integration?
'''

import torch

def get_camera_rays(H,W,f):
    '''
    Arguments:
    ----------
    (H,W): height and width of training images in pixels
    f: focal length (pixels) of the camera used

    Returns:
    --------
    cam_rays: (H,W,3) 3d rays created from camera spaced by focal length

    rays follow pinhole model
    all rays point into z=1 plane in projective geometry coordinates
    '''

    xx,yy = torch.meshgrid(torch.arange(H),torch.arange(W))

    #recenter using intrinsics of the camera
    # [f, 0, W/2]
    # [0, f, H/2]
    # [0, 0,   1]
    
    # move from pixel frame to camera frame (also in projective coords [x,y,1])
    x_cf = (xx-W/2)/f
    y_cf = (yy-H/2)/f
    cam_rays = torch.stack((x_cf,y_cf,torch.ones((H,W))), -1)
    return cam_rays

def get_nerf_rays(cam_rays, pose_cam):
    '''
    Arguments:
    ----------
    cam_rays: (H,W,3) camera rays created from get_camera_rays that represents the rays in camera coordinates
    pose_cam: (3,4) camera pose in the world coordinate
                    represents [3x3 rotation] + [1x3 translation]
                    can also be seen as the transformation from camera frame to world frame

    Returns:
    --------
    start_rays: (H*W,3) ray world position in xyz
    dir_rays: (H*W,3) ray normalized viewing direction

    NOTE: these are the rays that you will plug into nerf model :>
    '''

    #transform camera rays into world frame
    H,W,_ = cam_rays.shape
    dir_rays = cam_rays @ pose_cam[:3,:3].T

    start_rays = pose_cam[:,3].expand((H,W))

    dir_rays = dir_rays / torch.norm(dir_rays,dim=-1,keepdim=True) #normalize viewing direction

    #reshape in order to be parsed by NeRF
    dir_rays = dir_rays.reshape(-1,3)
    start_rays = start_rays.reshape(-1,3)

    return start_rays, dir_rays


if __name__ == '__main__':
    '''
        Unittest Ray Helper Functions: 
    '''

    print('Unittest Ray Helper Functions')
    print('=============================\n')

    