from keras import backend as K
import tensorflow as tf
import numpy as np

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, max_val=1, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = max_val  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ssim011(img1, img2, max_val=1, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = max_val  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    value = (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def cycleColoringLoss_SSIM011(y_true, y_pred):
    y_true=y_true[:,10:-10,10:-10,0:6]
    y_pred=y_pred[:,10:-10,10:-10,0:6]
    y_true_V=y_true[:,:,:,0:1]
    y_true_U=y_true[:,:,:,1:2]
    y_true_Y=y_true[:,:,:,2:3]
    y_true_reverse_V=y_true[:,:,:,3:4]
    y_true_reverse_U=y_true[:,:,:,4:5]
    y_true_reverse_Y=y_true[:,:,:,5:6]

    y_pred_V=y_pred[:,:,:,0:1]
    y_pred_U=y_pred[:,:,:,1:2]
    y_pred_Y=y_pred[:,:,:,2:3]
    y_pred_reverse_V=y_pred[:,:,:,3:4]
    y_pred_reverse_U=y_pred[:,:,:,4:5]
    y_pred_reverse_Y=y_pred[:,:,:,5:6]

    ssim1 = tf_ssim011(y_pred_Y, y_true_Y, max_val=255.0)
    ssim2 = tf_ssim(y_pred_reverse_V, y_true_reverse_V, max_val=255.0)
    ssim3 = tf_ssim(y_pred_reverse_U, y_true_reverse_U, max_val=255.0)

    ssim=(ssim1+ssim2+ssim3)/3.0
    return 1-ssim


def cycleColoringLoss_SSIM100111(y_true, y_pred):
    y_true=y_true[:,10:-10,10:-10,0:6]
    y_pred=y_pred[:,10:-10,10:-10,0:6]
    y_true_V=y_true[:,:,:,0:1]
    y_true_U=y_true[:,:,:,1:2]
    y_true_Y=y_true[:,:,:,2:3]
    y_true_reverse_V=y_true[:,:,:,3:4]
    y_true_reverse_U=y_true[:,:,:,4:5]
    y_true_reverse_Y=y_true[:,:,:,5:6]

    y_pred_V=y_pred[:,:,:,0:1]
    y_pred_U=y_pred[:,:,:,1:2]
    y_pred_Y=y_pred[:,:,:,2:3]
    y_pred_reverse_V=y_pred[:,:,:,3:4]
    y_pred_reverse_U=y_pred[:,:,:,4:5]
    y_pred_reverse_Y=y_pred[:,:,:,5:6]

    ssim1 = tf_ssim(y_pred_Y, y_true_Y, max_val=255.0)
    ssim2 = tf_ssim(y_pred_reverse_V, y_true_reverse_V, max_val=255.0)
    ssim3 = tf_ssim(y_pred_reverse_U, y_true_reverse_U, max_val=255.0)
    ssim4 = tf_ssim(y_pred_reverse_Y, y_true_reverse_Y, max_val=255.0)

    ssim=(ssim1+ssim2+ssim3+ssim4)/4.0
    return 1-ssim


def cycleColoringLoss_MSE100011(y_true, y_pred):
    y_true=y_true[:,10:-10,10:-10,0:6]
    y_pred=y_pred[:,10:-10,10:-10,0:6]
    y_true_V=y_true[:,:,:,0:1]
    y_true_U=y_true[:,:,:,1:2]
    y_true_Y=y_true[:,:,:,2:3]
    y_true_reverse_V=y_true[:,:,:,3:4]
    y_true_reverse_U=y_true[:,:,:,4:5]
    y_true_reverse_Y=y_true[:,:,:,5:6]

    y_pred_V=y_pred[:,:,:,0:1]
    y_pred_U=y_pred[:,:,:,1:2]
    y_pred_Y=y_pred[:,:,:,2:3]
    y_pred_reverse_V=y_pred[:,:,:,3:4]
    y_pred_reverse_U=y_pred[:,:,:,4:5]
    y_pred_reverse_Y=y_pred[:,:,:,5:6]

    MSE1 = K.mean(K.square(y_pred_Y - y_true_Y), axis=-1)
    MSE2 = K.mean(K.square(y_pred_reverse_V - y_true_reverse_V), axis=-1)
    MSE3 = K.mean(K.square(y_pred_reverse_U - y_true_reverse_U), axis=-1)

    MSE_val=(MSE1+MSE2+MSE3)/3.0
    return MSE_val


def cycleColoringLoss_MSE100111(y_true, y_pred):
    y_true=y_true[:,10:-10,10:-10,0:6]
    y_pred=y_pred[:,10:-10,10:-10,0:6]
    y_true_V=y_true[:,:,:,0:1]
    y_true_U=y_true[:,:,:,1:2]
    y_true_Y=y_true[:,:,:,2:3]
    y_true_reverse_V=y_true[:,:,:,3:4]
    y_true_reverse_U=y_true[:,:,:,4:5]
    y_true_reverse_Y=y_true[:,:,:,5:6]

    y_pred_V=y_pred[:,:,:,0:1]
    y_pred_U=y_pred[:,:,:,1:2]
    y_pred_Y=y_pred[:,:,:,2:3]
    y_pred_reverse_V=y_pred[:,:,:,3:4]
    y_pred_reverse_U=y_pred[:,:,:,4:5]
    y_pred_reverse_Y=y_pred[:,:,:,5:6]

    MSE1 = K.mean(K.square(y_pred_Y - y_true_Y), axis=-1)
    MSE2 = K.mean(K.square(y_pred_reverse_V - y_true_reverse_V), axis=-1)
    MSE3 = K.mean(K.square(y_pred_reverse_U - y_true_reverse_U), axis=-1)
    MSE4 = K.mean(K.square(y_pred_reverse_Y - y_true_reverse_Y), axis=-1)

    MSE_val=(MSE1+MSE2+MSE3+MSE4)/4.0
    return MSE_val