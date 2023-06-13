import numpy as np
import cv2
import g2o
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform

class Camera:
    def __init__(self, K):
        self.fx = K[0,0]
        self.fy = K[1,1]
        self.cx = K[0,2]
        self.cy = K[1,2]
        self.baseline = 0

def siftMatching(img1, img2):
    # Input : image1 and image2 in opencv format
    # Output : corresponding keypoints for source and target images
    # Output Format : Numpy matrix of shape: [No. of Correspondences X 2] 

    surf = cv2.SIFT_create()

    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Lowe's Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

    # Ransac
    model, inliers = ransac(
            (src_pts, dst_pts),
            AffineTransform, min_samples=4,
            residual_threshold=8, max_trials=10000
        )

    n_inliers = np.sum(inliers)

    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
    image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)


    src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
    dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

    return src_pts, dst_pts

def triangulate(ptsL, ptsR, K, poseL, poseR):
    '''
    '''
    
    camL_R = poseL[:3,:3]
    camL_t = poseL[:3,-1].reshape(3,1)
    camR_R = poseR[:3,:3]
    camR_t = poseR[:3,-1].reshape(3,1)
    
    
    qL = ptsL.copy()
    qR = ptsR.copy()
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    c = np.array([[cx,cy]]).T
    
    #setup the equations.
    qL = qL - c
    qL[0,:] = qL[0,:] / fx
    qL[1,:] = qL[1,:] / fy
    
    qR = qR - c
    qR[0,:] = qR[0,:] / fx
    qR[1,:] = qR[1,:] / fy
    
    qL = np.vstack( (qL,np.ones(qL.shape[1])) )
    qR = np.vstack( (qR,np.ones(qR.shape[1])) )
    
    Rlql = camL_R @ qL
    Rrqr = (camR_R @ qR) * -1
    b = camR_t - camL_t
    
    #formulate optimization problem
    zhatL = np.zeros(shape=(1,ptsL.shape[1]))
    zhatR = np.zeros(shape=(1,ptsR.shape[1]))
    for i in range(ptsL.shape[1]):
        R_lidx = Rlql[:,i].reshape(3,1)
        R_ridx = Rrqr[:,i].reshape(3,1)
        A = np.hstack((R_lidx,R_ridx))
        
        zhat = np.linalg.lstsq(A,b,rcond=None)[0]
        zhatL[:,i] = zhat[0]
        zhatR[:,i] = zhat[1]
    #afterwards, use the solution to solve for the points.
    PL = zhatL * qL
    PR = zhatR * qR
    
    P1 = camL_R @ PL + camL_t
    P2 = camR_R @ PR + camR_t
    pts3 = (P1+P2)/2
    return pts3

'''
    Bundle Adjustment Implementation using g2o
    ===========================================
        -> node: pose
        -> edge: ponti match
        -> unary factor: 2d pixel
'''
class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, rot, t, cam, fixed=False):
        sbacam = g2o.SBACam(rot, t)
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3) 

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()