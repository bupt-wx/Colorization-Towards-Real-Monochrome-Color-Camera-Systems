import numpy as np
import cv2
import os
import glob
import random
import math

def extractAllImage(lefts, rights):

        left_images = sorted(glob.glob(lefts + "/*.png"))
        right_images = sorted(glob.glob(rights + "/*.png"))
        return left_images, right_images

def splitData(l, r, val_ratio, fraction = 1):
	#tmp = zip(l, r)
	tmp = [(lhs, rhs) for lhs, rhs in zip(l, r)]
	random.shuffle(tmp)
	num_samples = len(l)
	num_data = int(fraction * num_samples)
	tmp = tmp[0:num_data]
	val_samples = int(math.ceil(num_data * val_ratio))
	val = tmp[0:val_samples]
	train = tmp[val_samples:]
	l_val, r_val = zip(*val)
	l_train, r_train = zip(*train)
	return [l_train, r_train], [l_val, r_val]
