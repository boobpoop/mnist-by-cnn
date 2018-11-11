import tensorflow as tf
import numpy as np

#step 2------define network structure
INPUT_NODE = 784
OUTPUT_NODE = 10

BATCH_SIZE = 100
IMAGE_SIZE = 28
INPUT_CHANNELS = 1
INPUT_LABLES = 10
CONV1_SIZE = 5
CONV1_DEEP = 6
CONV2_SIZE = 5
CONV2_DEEP = 16
FC1_SIZE = 120

#BATCH_SIZE = 100
#IMAGE_SIZE = 28
#INPUT_CHANNELS = 1
#INPUT_LABLES = 10
#CONV1_SIZE = 5
#CONV1_DEEP = 32
#CONV2_SIZE = 5
#CONV2_DEEP = 64
#FC1_SIZE = 512


def get_weight(shape, regularizer):
    weight = tf.get_variable("weight", shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weight))
    return weight

def get_bias(shape):
    bias = tf.get_variable("bias", shape, initializer = tf.constant_initializer(0.1))
    return bias

def conv2d(x, w, stride):
    conv = tf.nn.conv2d(x, w, strides = [1, stride, stride, 1], padding = "SAME")
    return conv

def max_pool(x, size, stride):
    pool = tf.nn.max_pool(x, ksize = [1, size, size, 1], strides = [1, stride, stride, 1], padding = "SAME")
    return pool

def forward_propagation(input_data_x, train,  regularizer):
    with tf.variable_scope("layer1--conv1"):
        input_x = tf.reshape(input_data_x, [-1 , IMAGE_SIZE, IMAGE_SIZE, INPUT_CHANNELS])
        conv1_weight = get_weight([CONV1_SIZE, CONV1_SIZE, INPUT_CHANNELS, CONV1_DEEP], None)   
        conv1_bias = get_bias([CONV1_DEEP])
        conv1 = conv2d(input_x, conv1_weight, 1)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
        print(relu1)

    with tf.variable_scope("layer2--pool1"):
        pool1 = max_pool(relu1, 2, 1)
        print(pool1)

    with tf.variable_scope("layer3--conv2"):
        conv2_weight = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], None)
        conv2_bias = get_bias([CONV2_DEEP])
        conv2 = conv2d(pool1, conv2_weight, 1)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
        print(relu2)

    with tf.variable_scope("layer4--pool2"):
        pool2 = max_pool(relu2, 2, 2)
        print(pool2)

    #dimension reduction
    pool_shape = pool2.get_shape().as_list()
    print(pool_shape)
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    print(nodes)
    reshaped = tf.reshape(pool2, [-1 , nodes])
    

    with tf.variable_scope("layer5--fc1"):
        fc1_weight = get_weight([nodes, FC1_SIZE], regularizer)
        fc1_bias = get_bias([FC1_SIZE])
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_bias)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

#    with tf.variable_scope("layer6--fc2"):
#        fc2_weight = get_weight([FC1_SIZE, FC2_SIZE], regularizer)
#        fc2_bias = get_bias([FC2_SIZE])
#        fc2 = tf.nn.relu(tf.matmul(fc1,  fc2_weight) + fc2_bias)
#        if train:
#            fc2 = tf.nn.dropout(fc2, 0.5)

    
    with tf.variable_scope("layer6--fc2"):
        fc3_weight = get_weight([FC1_SIZE, OUTPUT_NODE], regularizer)
        fc3_bias = get_bias([OUTPUT_NODE])
        y = tf.nn.relu(tf.matmul(fc1, fc3_weight) + fc3_bias)
    return y
