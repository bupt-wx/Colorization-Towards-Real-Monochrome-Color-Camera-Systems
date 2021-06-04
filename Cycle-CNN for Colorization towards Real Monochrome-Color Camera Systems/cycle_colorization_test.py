import sys
sys.path.append('src')
import numpy as np
import argparse
import parse_arguments
from coloringnetwork import *
import glob
import os
import psutil
from generator import *
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def _rotateImage_(img):
    (h, w) = img.shape[:2]
    center=(w/2-0.5,h/2-0.5)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def Predict():
    hp, tp, up, env = parse_arguments.parseArguments()

    parser = argparse.ArgumentParser()
    parser.add_argument('-fpath', help = 'file path of the dataset', required = True, default = None)
    parser.add_argument('-outpath', help = 'file output path of the dataset', required = True, default = None)

    parser.add_argument('-bs', help = 'batch size or steps', default = tp['batch_size'])
    args = parser.parse_args()
    outpath = args.outpath
    ext = up['file_extension']
    bs = tp['batch_size']
    file_path = args.fpath
    max_q_size = tp['max_q_size']
    verbose = tp['verbose']

    def get_session(gpu_fraction=0.95):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(get_session())
    model = createCycleColoringNetwork(hp, True)

    left_path = file_path + "/mono_5"
    right_path = file_path + "/color_5"
    left_images = glob.glob(left_path + "/*.{}".format(ext))
    right_images = glob.glob(right_path + "/*.{}".format(ext))
    
    data_path_test_out = outpath + "/"

    list_len=len(left_images)
    print('left_images length',list_len)

    for i in range(0, list_len):
        print(i,left_images[i])
        left_image=[left_images[i]]
        right_image=[right_images[i]]

        print('left image length',len(left_image))
        is_train = False
        generator = generate_arrays_from_file(left_image, right_image, up, is_train)
        print('generator is',generator)
        print("Predict data using generator...")
        pred = model.predict_generator(generator, max_queue_size = max_q_size, steps = bs, verbose = verbose)
        pred = pred[0,:,:,:]

        print(left_image[0].split('/')[-1].split('.')[0])
        cur_png_path = data_path_test_out + left_image[0].split('/')[-1]
        print(cur_png_path)

        pred_normal = pred[:,:,0:3]
        pred_reverse = pred[:,:,3:6]
        pred_normal=_rotateImage_(pred_normal)
        pred_reverse=_rotateImage_(pred_reverse)

        Left_image = cv2.imread(left_images[i])
        Right_image = cv2.imread(right_images[i])
        pred_normal[:, :, 2:3] = Left_image[:, :, 2:3]
        pred_reverse[:, :, 2:3] = Right_image[:, :, 2:3]

        cur_png_path_normal = data_path_test_out + 'normal/' + left_image[0].split('/')[-1]
        cur_png_path_reverse = data_path_test_out + 'reverse/' + left_image[0].split('/')[-1]
        print(cur_png_path_normal)
        print(cur_png_path_reverse)
        
        f=cv2.imwrite(cur_png_path_normal, pred_normal)
        f=cv2.imwrite(cur_png_path_reverse, pred_reverse)

    print("Testing Completed")
    K.clear_session()

if __name__ == "__main__":
    Predict()   
