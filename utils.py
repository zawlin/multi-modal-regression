import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from pyquaternion import Quaternion
import math
import scipy

def debug():
    import ipdb; ipdb.set_trace()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def drawbb(im,bbox_v2d):
    bbox_v2d=[tuple(v) for v in bbox_v2d]
    for i in range(len(bbox_v2d)):
        cv2.putText(im,'%d'%i, bbox_v2d[i], cv2.FONT_HERSHEY_SIMPLEX, 1, 255,2)
    cf = (255,0,0)
    cb = (255,255,0)
    cc = (0,255,0)
    cv2.line(im,bbox_v2d[0],bbox_v2d[1],cf,2)
    cv2.line(im,bbox_v2d[1],bbox_v2d[3],cf,2)
    cv2.line(im,bbox_v2d[2],bbox_v2d[3],cf,2)
    cv2.line(im,bbox_v2d[2],bbox_v2d[0],cf,2)

    cv2.line(im,bbox_v2d[4],bbox_v2d[5],cb,2)
    cv2.line(im,bbox_v2d[5],bbox_v2d[7],cb,2)
    cv2.line(im,bbox_v2d[6],bbox_v2d[7],cb,2)
    cv2.line(im,bbox_v2d[6],bbox_v2d[4],cb,2)

    cv2.line(im,bbox_v2d[4],bbox_v2d[0],cc,2)
    cv2.line(im,bbox_v2d[6],bbox_v2d[2],cc,2)
    cv2.line(im,bbox_v2d[5],bbox_v2d[1],cc,2)
    cv2.line(im,bbox_v2d[3],bbox_v2d[7],cc,2)

def draw_mesh(im,v2d_orig,faces):
    im_draw = im.copy()
    cc = (255,0,0)
    v2d = [tuple(v.astype(np.int32)) for v in v2d_orig]
    for f in faces:
        cv2.line(im_draw,v2d[f[0]],v2d[f[1]],cc,1)
        cv2.line(im_draw,v2d[f[1]],v2d[f[2]],cc,1)
        cv2.line(im_draw,v2d[f[2]],v2d[f[0]],cc,1)
    im[...] = (.3*im.astype(np.float32)+.7*im_draw.astype(np.float32)).astype(np.uint8)


def draw_mesh_poly(im,v2d_orig,faces,cc=(255,0,0)):
    im_draw = im.copy()
    v2d = [tuple(v.astype(np.int32)) for v in v2d_orig]
    for f in faces:
        pts = np.array([[v2d[f[0]],v2d[f[1]],v2d[f[2]]]],dtype=np.int32)
        cv2.fillPoly(im_draw,pts,cc)
        # cv2.line(im_draw,v2d[f[0]],v2d[f[1]],cc,1)
        # cv2.line(im_draw,v2d[f[1]],v2d[f[2]],cc,1)
        # cv2.line(im_draw,v2d[f[2]],v2d[f[0]],cc,1)
    im[...] = (.3*im.astype(np.float32)+.7*im_draw.astype(np.float32)).astype(np.uint8)
def plt_imshow(im_arr,fig_sz=(3,3)):
    f = plt.figure(figsize=fig_sz)
    sz = len(im_arr)
    plt.axis('off')
    for i in range(sz):
        ax1 = f.add_subplot(1,sz,i+1)
        plt.axis('off')
        ax1.imshow(im_arr[i])
    plt.tight_layout()
    plt.show()


def load(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def save(path,obj):
    with open(path,'wb') as f:
        pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
_time_last = 0
def tick():
    global _time_last
    _time_last=time.time()

def tock():
    return time.time()-_time_last

def pnp(objectPoints,imgPoints,w,h):

    f = 3000
    cameraMatrix = np.array([[f,0,w/2.0],
                             [0,f,h/2.0],
                             [0,0,1]])
    distCoeffs = np.zeros((5,1))
    revtval,rvecs, tvecs  =cv2.solvePnP(objectPoints[:,np.newaxis,:], imgPoints[:,np.newaxis,:], cameraMatrix, distCoeffs)#,False,flags=cv2.SOLVEPNP_EPNP)

    return rvecs,tvecs
    # imgpts_bb, jac = cv2.projectPoints(objectPoints, rvecs, tvecs, cameraMatrix, distCoeffs)
    # imgpts, jac = cv2.projectPoints(meshPoints, rvecs, tvecs, cameraMatrix, distCoeffs)
    # imgpts_bb= imgpts_bb.squeeze().astype(np.int32)
    # imgpts= imgpts.squeeze().astype(np.int32)

def project(objectPoints,w,h,rvecs,tvecs):
    f = 3000
    cameraMatrix = np.array([[f,0,w/2.0],
                             [0,f,h/2.0],
                             [0,0,1]])
    distCoeffs = np.zeros((5,1))

    imgpts, jac = cv2.projectPoints(objectPoints, rvecs, tvecs, cameraMatrix, distCoeffs)

    return imgpts

def rot_params_rv(rvecs):
    from math import pi,atan2,asin
    R = cv2.Rodrigues(rvecs)[0]
    roll = 180*atan2(-R[2][1], R[2][2])/pi
    pitch = 180*asin(R[2][0])/pi
    yaw = 180*atan2(-R[1][0], R[0][0])/pi
    rot_params= [roll,pitch,yaw]
    return rot_params


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rot_params_q(q):
    from math import pi,atan2,asin
    #R = q.rotation_matrix
    #R = rotation_matrix(q.axis,q.angle)
    R = q.rotation_matrix
    roll = 180*atan2(-R[2][1], R[2][2])/pi
    pitch = 180*asin(R[2][0])/pi
    yaw = 180*atan2(-R[1][0], R[0][0])/pi
    rot_params= [roll,pitch,yaw]
    return rot_params

def rot_params_r(R):
    from math import pi,atan2,asin
    #R = q.rotation_matrix
    #R = rotation_matrix(q.axis,q.angle)
    roll = 180*atan2(-R[2][1], R[2][2])/pi
    pitch = 180*asin(R[2][0])/pi
    yaw = 180*atan2(-R[1][0], R[0][0])/pi
    rot_params= [roll,pitch,yaw]
    return rot_params


def toq(rvecs):
    R = cv2.Rodrigues(rvecs)[0]
    return  Quaternion(matrix = R)

def load_quats(quat_path):
    text_file = open(quat_path,'r')
    lines = text_file.readlines()
    qa_nps = []
    qas = []
    for l in lines:
        ln = l.split('\t')
        ln = [float(li) for li in ln]
        qa_np = np.array([ln[3],ln[0],ln[1],ln[2]])#input in x,yz,w
        qa= Quaternion(qa_np)
        qa_nps.append(qa_np)
        qas.append(qa)
    return np.array(qa_nps),qas

def eulerAnglesToRotationMatrix(theta_degree) :
    theta = -(np.array(theta_degree)*math.pi/180)
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def gd(q1,q2):
    r1 = q1.rotation_matrix
    r2 = q2.rotation_matrix
    return np.linalg.norm(scipy.linalg.logm(np.matmul(r1.T,r2)))/np.sqrt(2)
