import cv2
import g2o
import numpy as np
import os
from collections import defaultdict
import json
from skimage.measure import ransac


import cv2
import numpy as np

from ba_utils import *
from scipy.spatial.transform import Rotation as R


if __name__ == '__main__':
    root = './'
    dataset = 'datanerf_static'
    PATH = os.path.join(root,dataset,'pv')
    image_paths = os.listdir(PATH)
    image_paths = sorted(image_paths)

    im_limit = 5

    ims = [cv2.imread(os.path.join(PATH,im_path))[:,:,::-1] for im_path in image_paths][:im_limit]

    im_names = [im_path.split('.')[0] for im_path in image_paths][:im_limit]

    with open(os.path.join(root,dataset,'pv_pose.json')) as f:
        j_pv = json.load(f)

    j_pv_refine = {}

    #create SIFT
    sift = cv2.SIFT_create()

    #hard-coded
    K = np.array([[907.17175,0,424/2],[0,906.81775,240/2],[0,0,1]])

    GraphOptimizer = BundleAdjustment()


    #add all of the poses
    for i in range(len(ims)):
        if i == 0:
            fixed = True
        else:
            fixed = False
        
        pose0 = np.array(j_pv[im_names[i]]).T
        Rmat0 = pose0[:3,:3]
        t0 = pose0[:3,-1]
        rot0 = g2o.g2opy.Quaternion(R.from_matrix(Rmat0).as_quat())  #quaternion

        GraphOptimizer.add_pose(i,rot0,t0,Camera(K),fixed=fixed)

    #add point nodes and edges
    for i in range(len(ims)-1):
        im0 = ims[i]
        im1 = ims[i+1]
        src_pts, dst_pts = siftMatching(im0,im1)
 
        pose0 = np.array(j_pv[im_names[i]]).T
        pose1 = np.array(j_pv[im_names[i+1]]).T

        pts_3d = triangulate(src_pts.T, dst_pts.T, K, pose0, pose1)

        N = pts_3d.shape[1]
        for j in range(N):
            GraphOptimizer.add_point(j,pts_3d[:,j])
            GraphOptimizer.add_edge(j,i,src_pts[j,:])
            GraphOptimizer.add_edge(j,i+1,dst_pts[j,:])
    
    print('Running Optimization')
    GraphOptimizer.optimize(max_iterations=100)
    print('Finished Optimizing')

    for i in range(len(ims)):
        R = GraphOptimizer.get_pose(i).orientation().rotation_matrix()    
        t = GraphOptimizer.get_pose(i).translation()

        mat = np.eye(4)
        mat[:3,:3] = R
        mat[:3,-1] = t

        j_pv_refine[im_names[i]] = mat.T.tolist()
        


    #save poses
    with open(os.path.join(root,dataset,'pv_pose_refine.json'), 'w') as f:
        json.dump(j_pv_refine, f)
    print('Done saving poses')