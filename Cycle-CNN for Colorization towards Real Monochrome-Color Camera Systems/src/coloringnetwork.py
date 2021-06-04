from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, Conv3D, Conv2DTranspose
from conv3dTranspose import Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras import backend as K
from keras.layers import Input, Add, add, multiply, AveragePooling2D, Dense
from keras.layers.core import Lambda, Permute, Reshape
from ipykernel import kernelapp as app
from spp.SpatialPyramidPooling import SpatialPyramidPooling
import tensorflow as tf
import numpy as np
import os

def _resNetBlock_(filters, ksize, stride, padding, act_func, name_head):
    conv1 = Conv2D(filters, ksize, strides = stride, padding = padding, name=name_head+'1')
    bn1 = BatchNormalization(axis = -1, name=name_head+'2')
    act1 = Activation(act_func, name=name_head+'3')
    conv2 = Conv2D(filters,ksize, strides = stride, padding = padding, name=name_head+'4')
    bn2 = BatchNormalization(axis = -1, name=name_head+'5')
    act2 = Activation(act_func, name=name_head+'6')
    add = Add()
    return [conv1, bn1, act1, conv2, bn2, act2, add]

def _addConv3D_(input, filters, ksize, stride, padding, name_head, bn = True, act_func = 'relu'):
    conv = Conv3D(filters, ksize, strides = stride, padding = padding, name=name_head+'1') (input)
    if bn:
        conv = BatchNormalization(axis = -1, name=name_head+'2')(conv)
    if act_func:
        conv = Activation(act_func, name=name_head+'3')(conv)
    return conv

def _convDownSampling_(input, filters, ksize, ds_stride, padding, name_head):
    conv = _addConv3D_(input, filters, ksize, ds_stride, padding,name_head+'conv1')
    conv = _addConv3D_(conv, filters, ksize, 1, padding,name_head+'conv2')    
    conv = _addConv3D_(conv, filters, ksize, 1, padding,name_head+'conv3')
    return conv

def _createDeconv3D_(input, filters, ksize, stride, padding, name_head, bn = True, act_func = 'relu'):
    deconv = Conv3DTranspose(filters, ksize, stride, padding, name=name_head+'1') (input)
    if bn:
        deconv = BatchNormalization(axis = -1, name=name_head+'2')(deconv)
    if act_func:
        deconv = Activation(act_func, name=name_head+'3')(deconv)
    return deconv

def _getFeatureVolume_(inputs, max_d):
    left_tensor, right_tensor = inputs
    shape = K.shape(right_tensor)
    right_tensor = K.spatial_2d_padding(right_tensor, padding=((0, 0), (max_d, 0)))
    disparity_costs = []
    for d in reversed(range(max_d)):
        left_tensor_slice = left_tensor
        right_tensor_slice = tf.slice(right_tensor, begin = [0, 0, d, 0], size = [-1, -1, shape[2], -1])
        cost = K.concatenate([left_tensor_slice, right_tensor_slice], axis = 3)
        disparity_costs.append(cost)
    feature_volume = K.stack(disparity_costs, axis = 1)
    return feature_volume

def _getReverseImage_(inputs):
    reverse_img = tf.image.flip_left_right(inputs)
    return reverse_img

def _concatImg_(inputs):
    result1, result2 = inputs
    output_tensor = K.concatenate([result1, result2], axis = 3)
    return output_tensor

def _getVUYData_(inputs):
    out = inputs[:,:,:,0:3]
    return out

def _getYData_(inputs):
    out = inputs[:,:,:,2:3]
    return out

def _getVUData_(inputs):
    out = inputs[:,:,:,0:2]
    return out

def _getGeoFeat_(inputs):
    out = inputs[:,:,:,3:5]
    return out

def _getWeightedAverage_(inputs, d):
    fv, right_img = inputs
    softmax_weights = tf.nn.softmax(fv, dim = 1)#1,d,h,w
    ref_V=right_img[:,:,:,0:1]
    ref_U=right_img[:,:,:,1:2]
    ref_Y=right_img[:,:,:,2:3]

    max_d = d
    right_tensor = ref_U[:]
    shape = K.shape(right_tensor)
    right_tensor = K.spatial_2d_padding(right_tensor, padding=((0, 0), (max_d, 0)))
    disparity_costs = []
    for d in reversed(range(max_d)):
        right_tensor_slice = tf.slice(right_tensor, begin = [0, 0, d, 0], size = [-1, -1, shape[2], -1])
        disparity_costs.append(right_tensor_slice)
    cost_volume = K.stack(disparity_costs, axis = 1)
    values = K.squeeze(cost_volume, axis=-1)
    c = tf.multiply(softmax_weights,values)
    U_map = tf.reduce_sum(c, axis = 1)

    right_tensor = ref_V[:]
    shape = K.shape(right_tensor)
    right_tensor = K.spatial_2d_padding(right_tensor, padding=((0, 0), (max_d, 0)))
    disparity_costs = []
    for d in reversed(range(max_d)):
        right_tensor_slice = tf.slice(right_tensor, begin = [0, 0, d, 0], size = [-1, -1, shape[2], -1])
        disparity_costs.append(right_tensor_slice)
    cost_volume = K.stack(disparity_costs, axis = 1)
    values = K.squeeze(cost_volume, axis=-1)
    c = tf.multiply(softmax_weights,values)
    V_map = tf.reduce_sum(c, axis = 1)

    right_tensor = ref_Y[:]
    shape = K.shape(right_tensor)
    right_tensor = K.spatial_2d_padding(right_tensor, padding=((0, 0), (max_d, 0)))
    disparity_costs = []
    for d in reversed(range(max_d)):
        right_tensor_slice = tf.slice(right_tensor, begin = [0, 0, d, 0], size = [-1, -1, shape[2], -1])
        disparity_costs.append(right_tensor_slice)
    cost_volume = K.stack(disparity_costs, axis = 1)
    values = K.squeeze(cost_volume, axis=-1)
    c = tf.multiply(softmax_weights,values)
    Y_map = tf.reduce_sum(c, axis = 1)

    VUY_map = tf.stack([V_map,U_map,Y_map], axis=3)
    return VUY_map


def _LearnReg_(input, base_num_filters, ksize, ds_stride, padding, num_down_conv):

    down_convs = list()
    name_head='Unet_conv1_'
    conv = _addConv3D_(input, base_num_filters, ksize, 1, padding, name_head) 
    name_head='Unet_conv2_'
    conv = _addConv3D_(conv, base_num_filters, ksize, 1, padding, name_head)

    down_convs.insert(0, conv)
    for i in range(num_down_conv):
        if i < num_down_conv - 1:
            mult = 2
        else:
            mult = 4
        name_head='Unet_down' + str(i)
        conv = _convDownSampling_(conv, mult * base_num_filters, ksize, ds_stride, padding, name_head)	 	
        down_convs.insert(0, conv)

    up_convs = down_convs[0]
    for i in range(num_down_conv):    
        filters = K.int_shape(down_convs[i+1])[-1]
        name_head='Unet_up' + str(i)
        deconv = _createDeconv3D_(up_convs, filters, ksize, ds_stride, padding, name_head) 
        up_convs = add([deconv, down_convs[i+1]])

    name_head='Unet_last_'
    conf_convs = _addConv3D_(up_convs, 1, ksize, 1, padding, name_head)
    cost = conf_convs
    return cost

def _createUniFeature_(input_shape, num_res, filters, first_ksize, ksize, act_func, padding):
    conv1 = Conv2D(filters, first_ksize, strides = 1, padding = padding, input_shape = input_shape, name='AF1')
    bn1 = BatchNormalization(axis = -1, name='AF2')
    act1 = Activation(act_func, name='AF3')
    conv2 = Conv2D(filters, ksize, strides = 1, padding = padding, name='AF4')
    bn2 = BatchNormalization(axis = -1, name='AF5')
    act2 = Activation(act_func, name='AF6')
    conv3 = Conv2D(filters, ksize, strides = 1, padding = padding, name='AF7')
    bn3 = BatchNormalization(axis = -1, name='AF8')
    act3 = Activation(act_func, name='AF9')
    conv4 = Conv2D(filters, ksize, strides = 1, padding = padding, name='AF10')
    bn4 = BatchNormalization(axis = -1, name='AF11')
    act4 = Activation(act_func, name='AF12')

    layers = [conv1, bn1, act1, conv2, bn2, act2, conv3, bn3, act3, conv4, bn4, act4]
    for i in range(num_res):
        name_head='AFres'+str(i)
        layers += _resNetBlock_(filters, ksize, 1, padding, act_func, name_head)
    return layers

def _processGeoFeature_(input_shape, filters, first_ksize, ksize, act_func, padding):

    conv1 = Conv2D(filters, first_ksize, strides = 1, padding = padding, input_shape = input_shape, name='GF1')
    bn1 = BatchNormalization(axis = -1, name='GF2')
    act1 = Activation(act_func, name='GF3')
    conv2 = Conv2D(filters, ksize, strides = 1, padding = padding, name='GF4')
    bn2 = BatchNormalization(axis = -1, name='GF5')
    act2 = Activation(act_func, name='GF6')
    conv3 = Conv2D(filters, ksize, strides = 1, padding = padding, name='GF7')
    bn3 = BatchNormalization(axis = -1, name='GF8')
    act3 = Activation(act_func, name='GF9')
    conv4 = Conv2D(2, ksize, strides = 1, padding = padding, name='GF10')
    bn4 = BatchNormalization(axis = -1, name='GF11')
    act4 = Activation(act_func, name='GF12')

    layers = [conv1, bn1, act1, conv2, bn2, act2, conv3, bn3, act3, conv4, bn4, act4]
    return layers

def createFeature(input, layers):
    tensor = input
    for layer in layers[0:12]:
        tensor = layer(tensor)
    res = tensor
    for layer in layers[12:]:
        if isinstance(layer, Add):
            tensor = layer([tensor, res])
            res = tensor
        else:
            tensor = layer(tensor)
    return tensor

def createCommonFeature(input, layers):
    tensor = input
    for layer in layers[0:]:
        tensor = layer(tensor)
    return tensor

def createCycleColoringNetwork(hp, pre_weight):

    d = hp['max_disp']
    first_ksize = hp['first_kernel_size']
    ksize = hp['kernel_size']
    num_filters = hp['base_num_filters']
    act_func = hp['act_func']
    num_down_conv = hp['num_down_conv']
    num_res = hp['num_res']
    ds_stride = hp['ds_stride']
    padding = hp['padding']
    K.set_image_data_format(hp['data_format'])

    input_shape = (None, None, 5)
    input_img_shape = (None, None, 3)
    input_geo_shape = (None, None, 2)
    left_input = Input(input_shape, dtype = "float32")
    right_input = Input(input_shape, dtype = "float32")

    left_img = Lambda(_getVUYData_, output_shape = input_img_shape)(left_input)
    right_img = Lambda(_getVUYData_, output_shape = input_img_shape)(right_input)

    left_geo_feat = Lambda(_getGeoFeat_, output_shape = (None, None, 2))(left_input)
    right_geo_feat = Lambda(_getGeoFeat_, output_shape = (None, None, 2))(right_input)

    app_layers = _createUniFeature_(input_img_shape, num_res, num_filters, first_ksize, ksize, act_func, padding)
    geo_layers = _processGeoFeature_(input_geo_shape, num_filters, first_ksize, ksize, act_func, padding)

    l_app_feature = createFeature(left_img, app_layers)
    r_app_feature = createFeature(right_img, app_layers)
    l_geo_feature = createCommonFeature(left_geo_feat,geo_layers)
    r_geo_feature = createCommonFeature(right_geo_feat,geo_layers)
    l_feature = Lambda(_concatImg_, output_shape = (None, None, num_filters+2))([l_app_feature,l_geo_feature])
    r_feature = Lambda(_concatImg_, output_shape = (None, None, num_filters+2))([r_app_feature,r_geo_feature])
    unifeatures = [l_feature, r_feature]

    fv = Lambda(_getFeatureVolume_, arguments = {'max_d':d}, output_shape = (d, None, None, (num_filters+2) * 2))(unifeatures)
    cv = _LearnReg_(fv, num_filters, ksize, ds_stride, padding, num_down_conv)
    wv = Lambda(K.squeeze, arguments = {'axis': -1}, output_shape = (d, None, None))(cv)
    unidata = [wv, right_img]

    colorization_result = Lambda(_getWeightedAverage_, arguments = {'d':d})(unidata)
    colorization_result_Y = Lambda(_getYData_, output_shape = (None, None, 1))(colorization_result)
    colorization_result_VU = Lambda(_getVUData_, output_shape = (None, None, 2))(colorization_result)
    
    output_result = Lambda(_concatImg_, output_shape = ( None, None, 3))([colorization_result_VU, colorization_result_Y])
    output_result = Lambda(_concatImg_, output_shape = ( None, None, 5))([output_result, left_geo_feat])

    cost_model = Model([left_input, right_input], output_result)

    result1 = cost_model([left_input, right_input])
    result1_reverse = Lambda(_getReverseImage_, output_shape = ( None, None, None))(result1)
    right_input_reverse = Lambda(_getReverseImage_, output_shape = ( None, None, None))(right_input)  
    result2 = cost_model([right_input_reverse , result1_reverse])  
    result2 = Lambda(_getReverseImage_, output_shape = ( None, None, None))(result2)
    result1_img = Lambda(_getVUYData_, output_shape = input_img_shape)(result1)
    result2_img = Lambda(_getVUYData_, output_shape = input_img_shape)(result2)
    uniresults = [result1_img, result2_img]

    output_img_volume = Lambda(_concatImg_, output_shape = ( None, None, None))(uniresults)
    cycle_model = Model([left_input , right_input], output_img_volume)

    if pre_weight == 1:
        # train attention net
        print("Loading pretrained cost weight...")
        cycle_model.load_weights('model/s0.h5', by_name=True)

    cycle_model.summary()
    return cycle_model

