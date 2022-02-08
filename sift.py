import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
import itertools as it
import matplotlib.pyplot as plt
import math
import copy
from numba import jit
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from utilsdir.sift_func import *
from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging

from sklearn.neighbors import KDTree



def get_nearest_points(p1,p2,tree):
    p2_new = np.empty(p1.shape)
    index = np.empty(p1.shape[0])
    for i,point in enumerate(p1):
        dist , ind = tree.query(p1[i,None],k=1)
        p2_new[i] = p2[ind]
        index[i] = ind
    return p2_new,index

if __name__ == "__main__":
    img1 = './images/cat.jpg'           
    img2 = './images/cat_r.jpg'

    img1_mat = cv2.imread(img1)
    img2_mat = cv2.imread(img2)


    # Compute SIFT keypoints and descriptors
    kp1, des1 = GetDescriptors(img1)
    kp2, des2 = GetDescriptors(img2)


    des1 = np.array(des1)
    des2 = np.array(des2) 

    # np.save('desc1.npy', des1)
    # np.save('desc2.npy', des2)
    # tree = KDTree(des2, leaf_size=2)
    # des2,index = get_nearest_points(des1,des2,tree)
    
    # index = np.array(index,dtype="int")
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    keypoints1 = []
    keypoints2 = []
    for k in kp1:
        keypoints1.append(k['keypoint'])
    for k in kp2:
        keypoints2.append(k['keypoint'])

    out = cv2.drawMatches(img1_mat,keypoints1,img2_mat,keypoints2,matches,img2_mat)
    cv2.imwrite('matches.jpg',out)
    plt.imshow(out)



