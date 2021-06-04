import cv2
import numpy as np
import glob
import random
from load_pfm import *
import tensorflow as tf

def generate_arrays_from_file(lefts, rights, up, is_train):

        train = is_train # True or False
        random.seed(up['seed'])

        while 1:
            for ldata, rdata in zip(lefts, rights):

                left_img = cv2.imread(ldata)
                right_img = cv2.imread(rdata)
                left_img_rotate = _rotateImage_(left_img)
                right_img_rotate = _rotateImage_(right_img)

                img_shape = left_img_rotate.shape
                left_geo_feat = _getGeometryFeat_(img_shape)
                right_geo_feat = _getGeometryFeat_(img_shape)

                left_img_rotate = _centerImage_(left_img_rotate)
                right_img_rotate = _centerImage_(right_img_rotate)
                left_geo_feat = _centerImage_(left_geo_feat)
                right_geo_feat = _centerImage_(right_geo_feat)
                
                left_img_rotate = np.expand_dims(left_img_rotate, 0)
                right_img_rotate = np.expand_dims(right_img_rotate, 0)
                left_geo_feat = np.expand_dims(left_geo_feat, 0)
                right_geo_feat = np.expand_dims(right_geo_feat, 0)
                
                left_input = np.concatenate([left_img_rotate, left_geo_feat], axis = 3)
                right_input = np.concatenate([right_img_rotate, right_geo_feat], axis = 3)

                if train == True:
                    VUY_map = np.concatenate((left_img_rotate,right_img_rotate),axis=3)
                    yield ([left_input, right_input], VUY_map)
                else:
                    yield ([left_input, right_input])

            if not train:
                break

def _centerImage_(img):
    img = img.astype(np.float32)
    return img

def _rotateImage_(img):
    (h, w) = img.shape[:2]
    center=(w/2-0.5,h/2-0.5)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def _getGeometryFeat_(img_shape):
    H = img_shape[0]
    W = img_shape[1]
    feat = np.zeros((H,W,2))
    for j in range(H):
        for i in range(W):
            feat[j,i,0]=np.min([j-0,H-1-j])/(H-1)*1.0            
            feat[j,i,1]=np.min([i-0,W-1-i])/(W-1)*1.0
    return feat