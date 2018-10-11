#coding=utf-8

'''
@author LiangYu
@email  liangyufz@gmail.com
@create date 2018-07-13 18:55:11
@modify date 2018-08-15 11:00:43
@desc [description]
'''
import math
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

timeformat2 = "[%m%d_%H%M]"     

class NetworkOps(object):
    """ Operations that are frequently used within networks. """
    neg_slope_of_relu = 0.01

    @classmethod
    def leaky_relu(cls, tensor, name='relu'):
        out_tensor = tf.maximum(tensor, cls.neg_slope_of_relu*tensor, name=name)
        return out_tensor

    @classmethod
    def conv(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            strides = [1, stride, stride, 1]
            kernel_shape = [kernel_size, kernel_size, in_size[3], out_chan]

            # conv
            kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                     tf.contrib.layers.xavier_initializer_conv2d(), trainable=trainable, collections=['wd', 'variables', 'filters'])
            tmp_result = tf.nn.conv2d(in_tensor, kernel, strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')

            return out_tensor

    @classmethod
    def conv_relu(cls, in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
        tensor = cls.conv(in_tensor, layer_name, kernel_size, stride, out_chan, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @classmethod
    def max_pool(cls, bottom, name='pool'):
        pooled = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='VALID', name=name)
        return pooled

    @classmethod
    def upconv(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()

            kernel_shape = [kernel_size, kernel_size, in_size[3], in_size[3]]
            strides = [1, stride, stride, 1]

            # conv
            kernel = cls.get_deconv_filter(kernel_shape, trainable)
            tmp_result = tf.nn.conv2d_transpose(value=in_tensor, filter=kernel, output_shape=output_shape,
                                                strides=strides, padding='SAME')

            # bias
            biases = tf.get_variable('biases', [kernel_shape[2]], tf.float32,
                                     tf.constant_initializer(0.0), trainable=trainable, collections=['wd', 'variables', 'biases'])
            out_tensor = tf.nn.bias_add(tmp_result, biases)
            return out_tensor

    @classmethod
    def upconv_relu(cls, in_tensor, layer_name, output_shape, kernel_size, stride, trainable=True):
        tensor = cls.upconv(in_tensor, layer_name, output_shape, kernel_size, stride, trainable)
        out_tensor = cls.leaky_relu(tensor, name='out')
        return out_tensor

    @staticmethod
    def get_deconv_filter(f_shape, trainable):
        width = f_shape[0]
        height = f_shape[1]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init,
                               shape=weights.shape, trainable=trainable, collections=['wd', 'variables', 'filters'])

    @staticmethod
    def fully_connected(in_tensor, layer_name, out_chan, trainable=True):
        with tf.variable_scope(layer_name):
            in_size = in_tensor.get_shape().as_list()
            assert len(in_size) == 2, 'Input to a fully connected layer must be a vector.'
            weights_shape = [in_size[1], out_chan]

            # weight matrix
            weights = tf.get_variable('weights', weights_shape, tf.float32,
                                      tf.contrib.layers.xavier_initializer(), trainable=trainable)
            weights = tf.check_numerics(weights, 'weights: %s' % layer_name)

            # bias
            biases = tf.get_variable('biases', [out_chan], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=trainable)
            biases = tf.check_numerics(biases, 'biases: %s' % layer_name)

            out_tensor = tf.matmul(in_tensor, weights) + biases
            return out_tensor

    @classmethod
    def fully_connected_relu(cls, in_tensor, layer_name, out_chan, trainable=True):
        tensor = cls.fully_connected(in_tensor, layer_name, out_chan, trainable)
        out_tensor = tf.maximum(tensor, cls.neg_slope_of_relu*tensor, name='out')
        return out_tensor

    @staticmethod
    def dropout(in_tensor, keep_prob, evaluation):
        """ Dropout: Each neuron is dropped independently. """
        with tf.variable_scope('dropout'):
            tensor_shape = in_tensor.get_shape().as_list()
            out_tensor = tf.cond(evaluation,
                                 lambda: tf.nn.dropout(in_tensor, 1.0,
                                                       noise_shape=tensor_shape),
                                 lambda: tf.nn.dropout(in_tensor, keep_prob,
                                                       noise_shape=tensor_shape))
            return out_tensor

    @staticmethod
    def spatial_dropout(in_tensor, keep_prob, evaluation):
        """ Spatial dropout: Not each neuron is dropped independently, but feature map wise. """
        with tf.variable_scope('spatial_dropout'):
            tensor_shape = in_tensor.get_shape().as_list()
            out_tensor = tf.cond(evaluation,
                                 lambda: tf.nn.dropout(in_tensor, 1.0,
                                                       noise_shape=tensor_shape),
                                 lambda: tf.nn.dropout(in_tensor, keep_prob,
                                                       noise_shape=[tensor_shape[0], 1, 1, tensor_shape[3]]))
            return out_tensor

class LearningRateScheduler:
    """
        Provides scalar tensors at certain iteration as is needed for a multistep learning rate schedule.
    """
    def __init__(self, steps, values):
        self.steps = steps
        self.values = values

        assert len(steps)+1 == len(values), "There must be one more element in value as step."

    def get_lr(self, global_step):
        with tf.name_scope('lr_scheduler'):

            if len(self.values) == 1: #1 value -> no step
                learning_rate = tf.constant(self.values[0])
            elif len(self.values) == 2: #2 values -> one step
                cond = tf.greater(global_step, self.steps[0])
                learning_rate = tf.where(cond, self.values[1], self.values[0])
            else: # n values -> n-1 steps
                cond_first = tf.less(global_step, self.steps[0])

                cond_between = list()
                for ind, _ in enumerate(range(0, len(self.steps)-1)):  # for ind , step
                    cond_between.append(tf.logical_and(tf.less(global_step, self.steps[ind+1]),
                                                       tf.greater_equal(global_step, self.steps[ind])))

                cond_last = tf.greater_equal(global_step, self.steps[-1])

                cond_full = [cond_first]
                cond_full.extend(cond_between)
                cond_full.append(cond_last)

                cond_vec = tf.stack(cond_full)
                lr_vec = tf.stack(self.values)

                learning_rate = tf.where(cond_vec, lr_vec, tf.zeros_like(lr_vec))

                learning_rate = tf.reduce_sum(learning_rate)

            return learning_rate



def load_weights_from_snapshot(session, checkpoint_path, discard_list=None, rename_dict=None):
    """Loads weights from a snapshot except the ones indicated with discard_list. Others are possibly renamed. """
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Remove everything from the discard list
    if discard_list is not None:
        num_disc = 0
        var_to_shape_map_new = dict()
        for k, v in var_to_shape_map.items():
            good = True
            for dis_str in discard_list:
                if dis_str in k:
                    good = False

            if good:
                var_to_shape_map_new[k] = v
            else:
                num_disc += 1
        var_to_shape_map = dict(var_to_shape_map_new)
        print(time.strftime(timeformat2) + 'Discarded %d items' % num_disc)

    # rename everything according to rename_dict
    num_rename = 0
    var_to_shape_map_new = dict()
    for name in var_to_shape_map.keys():
        new_name = name
        if rename_dict is not None:
            for rename_str in rename_dict.keys():
                if rename_str in name:
                    new_name = new_name.replace(rename_str, rename_dict[rename_str])
                    num_rename += 1
        var_to_shape_map_new[new_name] = reader.get_tensor(name)
    var_to_shape_map = dict(var_to_shape_map_new)

    init_op, init_feed = tf.contrib.framework.assign_from_values(var_to_shape_map)
    session.run(init_op, init_feed)
    print(time.strftime(timeformat2) + 'Initialized %d variables from %s.' % (len(var_to_shape_map), checkpoint_path))

def softmax_heatmap(scoremap):
    ss = scoremap.shape
    scoremap_flat_heatmap = tf.reshape(scoremap, (-1, 1, ss[1]*ss[2], ss[3]))
    softmax_scoremap_flat_heatmap = tf.nn.softmax(scoremap_flat_heatmap, dim=2)
    recover_scoremap = tf.reshape(softmax_scoremap_flat_heatmap, (-1, ss[1], ss[2], ss[3]))
    return recover_scoremap
