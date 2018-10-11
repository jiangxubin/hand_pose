#coding=utf-8

'''
@author LiangYu
@email  liangyufz@gmail.com
@create date 2018-07-18 10:00:00
@modify date 2018-08-15 11:02:34
@desc [description]
'''
import json
import numpy as np
import tensorflow as tf
def load_json(path):
    with open(path, encoding='utf-8') as f:
        result = json.load(f)
    return result

def store_json(filename, data):
    with open(filename, 'w') as json_file:
        json_file.write(json.dumps(data))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def convert_kp(keypoints):
    """ Maps the keypoints into the right order. """

    # mapping into my keypoint definition
    kp_dict = {0: 0, 1: 20, 2: 19, 3: 18, 4: 17, 5: 16, 6: 15, 7: 14, 8: 13, 9: 12, 10: 11, 11: 10,
               12: 9, 13: 8, 14: 7, 15: 6, 16: 5, 17: 4, 18: 3, 19: 2, 20: 1}

    keypoints_new = list()
    for i in range(21):
        if i in kp_dict.keys():
            pos = kp_dict[i]
            keypoints_new.append(keypoints[pos, :])

    return np.stack(keypoints_new, 0)
    