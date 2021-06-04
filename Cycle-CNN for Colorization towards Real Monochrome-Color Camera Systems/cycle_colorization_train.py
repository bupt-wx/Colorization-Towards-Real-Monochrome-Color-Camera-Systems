import argparse
import sys
sys.path.append('src')
from parse_arguments import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from data_utils import *
from custom_callback import customModelCheckpoint
import coloringnetwork
from generator import *
from losses import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import optimizers
from keras import backend as K
import math
import tensorflow as tf

def trainSceneFlowData(hp, tp, up, env, callbacks, upw):

        lr = tp['learning_rate']
        epochs = tp['epochs']
        batch_size = tp['batch_size']
        q_size = tp['max_q_size']
        epsilon = tp['epsilon']
        rho = tp['rho']
        decay = tp['decay']

        val_ratio = up['val_ratio']
        fraction = up['fraction']

        frank_root_path = env['frank_root']
        left_root = frank_root_path + '/mono_5'
        right_root = frank_root_path + '/color_5'

        left_imgs, right_imgs = extractAllImage(left_root, right_root)
        train, val = splitData(left_imgs, right_imgs, val_ratio, fraction)
        is_train = True
        val_generator = generate_arrays_from_file(val[0], val[1], up, is_train)
        train_generator = generate_arrays_from_file(train[0], train[1], up, is_train)

        num_steps = math.ceil(len(train[0]) / batch_size)
        val_steps = math.ceil(len(val[0]) / batch_size)

        num_steps = num_steps
        val_steps = val_steps

        model = coloringnetwork.createCycleColoringNetwork(hp, upw)
        optimizer = optimizers.RMSprop(lr = lr, rho = rho, epsilon = epsilon, decay = decay)
        model.compile(optimizer = optimizer, loss = cycleColoringLoss_SSIM011, metrics = [cycleColoringLoss_SSIM011,cycleColoringLoss_SSIM100111,cycleColoringLoss_MSE100011,cycleColoringLoss_MSE100111])
        model.fit_generator(train_generator, validation_data = val_generator, validation_steps = val_steps, steps_per_epoch = num_steps, max_q_size = q_size, epochs = epochs,  callbacks = callbacks)
        print("Training Complete")
        model.save_weights('model/s0.h5')
        result = model.predict_generator(train_generator, steps = 1)
        print(result)
        print(result.shape)
        np.save("prediction.npy", result)


def genCallBacks(cost_filepath, outputfilepath, log_save_path, save_best_only, period, verbose):
        callback_tb = TensorBoard(log_dir = log_save_path, histogram_freq = 0, write_graph = True, write_images = True)
        callback_mc = customModelCheckpoint(cost_filepath, outputfilepath, verbose = verbose, save_best_only = save_best_only, period = period)
        return [callback_tb, callback_mc]


if __name__ == '__main__':

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.gpu_options.allocator_type ='BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.98
        sess = tf.Session(config = config)
        K.set_session(sess)
        
        hp, tp, up, env = parseArguments()
        parser = argparse.ArgumentParser()
        parser.add_argument('-upw', '--use_pretrained_weight', type = int, help = 'train the model use pretrained weight', default = 1)
        args = parser.parse_args()

        log_save_path = tp['log_save_path']
        save_best_only = tp['save_best_only']
        period = tp['period']
        verbose = tp['verbose']
        cost_weight_path = tp['cost_volume_weight_save_path']
        linear_output_weight_path = tp['linear_output_weight_path']

        if hp['output'] == 'softargmin':
                linear_output_weight_path = None

        callbacks = genCallBacks(cost_weight_path, linear_output_weight_path, log_save_path, save_best_only, period, verbose)
        trainSceneFlowData(hp, tp, up, env, callbacks, args.use_pretrained_weight)
