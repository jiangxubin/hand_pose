#coding=utf-8

'''
@author LiangYu
@email  liangyufz@gmail.com
@create date 2018-08-07 09:47:10
@modify date 2018-08-15 11:03:35
@desc [description]
'''

import os
import time
import tensorflow as tf
from utils.model_utils import NetworkOps, softmax_heatmap
from .dsnt import dsnt

timeformat2 = "[%m%d_%H%M]" 
ops = NetworkOps

class PoseNet(object):
    def __init__(self):
        self.crop_size = 256
        self.num_kp = 21
        
    def init(self, session, weight_files=None, exclude_var_list=None):
        """ Initializes weights from pickled python dictionaries.

            Inputs:
                session: tf.Session, Tensorflow session object containing the network graph
                weight_files: list of str, Paths to the pickle files that are used to initialize network weights
                exclude_var_list: list of str, Weights that should not be loaded
        """
        if exclude_var_list is None:
            exclude_var_list = list()

        import pickle

        if weight_files is None:
            weight_files = ['./weights/handsegnet-rhd.pickle', './weights/posenet3d-rhd-stb-slr-finetuned.pickle']

        # Initialize with weights
        for file_name in weight_files:
            assert os.path.exists(file_name), "File not found."
            with open(file_name, 'rb') as fi:
                weight_dict = pickle.load(fi)
                weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in exclude_var_list])}
                if len(weight_dict) > 0:
                    init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
                    session.run(init_op, init_feed)
                    print(time.strftime(timeformat2) + 'Loaded %d variables from %s' % (len(weight_dict), file_name))




    def inference(self, image_crop, train=False):
        """ PoseNet: Given an image it detects the 2D hand keypoints.
            The image should already contain a rather tightly cropped hand.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted
                train: bool, True in case weights should be trainable

            Outputs:
                scoremap_list_large: list of [B, 256, 256, 21] tf.float32 tensor, Scores for the hand keypoints
        """
        with tf.variable_scope('PoseNet2D'):
            scoremap_list = list()
            layers_per_block = [2, 2, 4, 2]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            #heatmap_dice = tf.get_variable("heatmap_dice", shape=(21,), trainable=train)

            # learn some feature representation, that describes the image content well
            x = image_crop
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            x = ops.conv_relu(x, 'conv4_3', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_4', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_5', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_6', kernel_size=3, stride=1, out_chan=256, trainable=train)
            encoding = ops.conv_relu(x, 'conv4_7', kernel_size=3, stride=1, out_chan=128, trainable=train)

            # use encoding to detect initial scoremap
            x = ops.conv_relu(encoding, 'conv5_1', kernel_size=1, stride=1, out_chan=512, trainable=train)
            scoremap = ops.conv(x, 'conv5_2', kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
            #scoremap = softmax_heatmap(scoremap*heatmap_dice)  #  before add to list ,mul dice and softmax it
            scoremap_list.append(scoremap)
            
            # iterate recurrent part a couple of times
            layers_per_recurrent_unit = 5
            num_recurrent_units = 2
            for pass_id in range(num_recurrent_units):
                x = tf.concat([scoremap_list[-1], encoding], 3)
                for rec_id in range(layers_per_recurrent_unit):
                    x = ops.conv_relu(x, 'conv%d_%d' % (pass_id+6, rec_id+1), kernel_size=7, stride=1, out_chan=128, trainable=train)
                x = ops.conv_relu(x, 'conv%d_6' % (pass_id+6), kernel_size=1, stride=1, out_chan=128, trainable=train)
                scoremap = ops.conv(x, 'conv%d_7' % (pass_id+6), kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
                #scoremap = softmax_heatmap(scoremap*heatmap_dice)  #  before add to list ,mul dice and softmax it
                scoremap_list.append(scoremap)

            scoremap_list_large = scoremap_list

            temp_scoremap = list()
            #for temp in scoremap_list_large:
            #    temp = temp*heatmap_dice
            #    softmax_tmp = softmax_heatmap(temp)
            #    temp_scoremap.append(softmax_tmp)

        #return scoremap_list_large, heatmap_dice, temp_scoremap
        return scoremap_list_large, "a", temp_scoremap
        #return scoremap_list_large


    def inference_with_dsnt(self, image_crop, train=False):
        """ PoseNet: Given an image it detects the 2D hand keypoints.
            The image should already contain a rather tightly cropped hand.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted
                train: bool, True in case weights should be trainable

            Outputs:
                scoremap_list_large: list of [B, 256, 256, 21] tf.float32 tensor, Scores for the hand keypoints
        """
        with tf.variable_scope('PoseNet2D'):
            scoremap_list = list()
            layers_per_block = [2, 2, 4, 2]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            # learn some feature representation, that describes the image content well
            x = image_crop
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            x = ops.conv_relu(x, 'conv4_3', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_4', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_5', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_6', kernel_size=3, stride=1, out_chan=256, trainable=train)
            encoding = ops.conv_relu(x, 'conv4_7', kernel_size=3, stride=1, out_chan=128, trainable=train)

            # use encoding to detect initial scoremap
            x = ops.conv_relu(encoding, 'conv5_1', kernel_size=1, stride=1, out_chan=512, trainable=train)
            scoremap = ops.conv(x, 'conv5_2', kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
            scoremap_list.append(scoremap)
            
            # iterate recurrent part a couple of times
            layers_per_recurrent_unit = 5
            num_recurrent_units = 2
            for pass_id in range(num_recurrent_units):
                x = tf.concat([scoremap_list[-1], encoding], 3)
                for rec_id in range(layers_per_recurrent_unit):
                    x = ops.conv_relu(x, 'conv%d_%d' % (pass_id+6, rec_id+1), kernel_size=7, stride=1, out_chan=128, trainable=train)
                x = ops.conv_relu(x, 'conv%d_6' % (pass_id+6), kernel_size=1, stride=1, out_chan=128, trainable=train)
                scoremap = ops.conv(x, 'conv%d_7' % (pass_id+6), kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
                scoremap_list.append(scoremap)

            heatmap_shape = scoremap_list[-1].shape
            dsnt_input = tf.reshape(tf.transpose(scoremap_list[-1], (0, 3, 1, 2)), (-1, heatmap_shape[1], heatmap_shape[2], 1), name="dsnt_input")
            norm_heatmap, coords_pred = dsnt(dsnt_input)

        return norm_heatmap, coords_pred, scoremap_list
        