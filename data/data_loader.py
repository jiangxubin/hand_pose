#coding=utf-8

'''
@author LiangYu
@email  liangyufz@gmail.com
@create date 2018-08-04 19:12:11
@modify date 2018-08-15 11:04:12
@desc [description]
'''

import time
import os
import fire
import ipdb

import tensorflow as tf
from config import opt

timeformat2 = "[%m%d_%H%M]" 

def create_dataset_from_TFRecords(filename, state="train", batch_size=64, is_shuffle=False, n_epochs=0):
    """create dataset for train and validation dataset"""
    assert os.path.exists(filename), "TFRecordsFile does not exist"
    print(time.strftime(timeformat2) + "Create dataset from ", filename)

    dataset = tf.data.TFRecordDataset(filename)

    if n_epochs > 1:
        dataset = dataset.repeat(n_epochs) # for train

    if state == "eval":
        setattr(opt, "augument", False)
        setattr(opt, "train_vis", False)
        setattr(opt, "need_heatmap_gt", False)
    if opt.with_dsnt:
        setattr(opt, "need_heatmap_gt", False)
        

    dataset = dataset.map(decode_and_process_TFRecords) # decode and prcess

    if is_shuffle:
        dataset = dataset.shuffle(1000 + 3 * batch_size) # shuffle  
    dataset = dataset.batch(batch_size)
    return dataset

### 方案一 使用文件队列 # 不适用
#def read_and_decodefilename, batch_size=1, shuffle=False, sigma=25.0):
    #filename_queue = tf.train.string_input_producer([filename],num_epochs=3)
    #reader = tf.TFRecordReader()
    #key, serialized_example = reader.read(filename_queue)
    #data_dict = dict()
    # ……
    
### 方案二 使用tf.data
def decode_TFRecords():
    pass



def decode_and_process_TFRecords(serialized_example):
    """decode the serialized example"""
    print(time.strftime(opt.timeformat2) + "corp_size:({opt.crop_size}) image_size_h/w:({opt.image_size_h}/{opt.image_size_w}) box_scale:({opt.box_scale}) sigma:({opt.sigma}) augument:({opt.augument}) train_vis:({opt.train_vis})".format(opt=opt))
    
    features = tf.parse_single_example(serialized_example, features={
        "image": tf.FixedLenFeature([], tf.string),
        "name": tf.FixedLenFeature([], tf.string),   
        "joints": tf.FixedLenFeature([], tf.string)})
    
    name = features['name']
    image = tf.decode_raw(features["image"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [opt.image_size_h, opt.image_size_w, 3])
    image = image / 255.0 - 0.5
    
    joints = tf.decode_raw(features["joints"], tf.float32)
    joints = tf.reshape(joints, [21, 3])

    ##### process

    keypoint_uv21 = joints[:, :2]
    keypoint_vis21 = tf.cast(joints[:, 2], tf.bool) 
    

    # crop center
    crop_center = keypoint_uv21[12, ::-1]
    
    # for train not for eval
    if opt.augument:
        noise = tf.truncated_normal([21, 2], mean=0.0, stddev=opt.coord_uv_noise_sigma)
        keypoint_uv21 += noise

        noise = tf.truncated_normal([2], mean=0.0, stddev=opt.crop_center_noise_sigma)
        crop_center += noise
        
    if opt.train_vis:
        keypoint_vis21 = tf.cast(tf.ones_like(keypoint_vis21), tf.bool)
        
    # select visible coords only
    kp_coord_h = tf.boolean_mask(keypoint_uv21[:, 1], keypoint_vis21)
    kp_coord_w = tf.boolean_mask(keypoint_uv21[:, 0], keypoint_vis21)
    kp_coord_hw = tf.stack([kp_coord_h, kp_coord_w], 1)

    # determine size of crop (measure spatial extend of hw coords first)
    min_coord = tf.maximum(tf.reduce_min(kp_coord_hw, 0), 0.0)
    max_coord = tf.minimum(tf.reduce_max(kp_coord_hw, 0), min(opt.image_size_h, opt.image_size_w))

    # find out larger distance wrt the center of crop
    crop_size_best = 2*tf.maximum(max_coord - crop_center, crop_center - min_coord)
    crop_size_best = tf.reduce_max(crop_size_best)*opt.box_scale
    crop_size_best = tf.minimum(tf.maximum(crop_size_best, 50.0), 500.0)
    
    # calculate necessary scaling
    scale = tf.cast(opt.crop_size, tf.float32) / crop_size_best
    scale = tf.minimum(tf.maximum(scale, 1.0), 10.0)
    
    # crop imag
    img_crop = crop_image_from_xy(tf.expand_dims(image, 0), crop_center, 256, scale)
    img_crop = img_crop[0]
    
    # Modify uv21 coordinates
    crop_center_float = tf.cast(crop_center, tf.float32)
    keypoint_uv21_u = (keypoint_uv21[:, 0] - crop_center_float[1]) * scale + opt.crop_size // 2
    keypoint_uv21_v = (keypoint_uv21[:, 1] - crop_center_float[0]) * scale + opt.crop_size // 2
    keypoint_uv21 = tf.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
    
    # create scoremaps from the subset of 2D annoataion
    keypoint_hw21 = tf.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], -1)
    scoremap_size = (opt.crop_size, opt.crop_size)
    if opt.need_heatmap_gt:
        scoremap = create_multiple_gaussian_map(keypoint_hw21, scoremap_size, opt.sigma, valid_vec=keypoint_vis21)
    else:
        scoremap = tf.zeros((opt.crop_size, opt.crop_size, 21)) 
    ##
    return img_crop, scoremap, keypoint_vis21, keypoint_uv21, scale, name


def crop_image_from_xy(image, crop_location, crop_size, scale=1.0):
    """
    Crops an image. When factor is not given does an central crop.

    Inputs:
        image: 4D tensor, [batch, height, width, channels] which will be cropped in height and width dimension
        crop_location: tensor, [batch, 2] which represent the height and width location of the crop
        crop_size: int, describes the extension of the crop
    Outputs:
        image_crop: 4D tensor, [batch, crop_size, crop_size, channels]
    """
    with tf.name_scope('crop_image_from_xy'):
        #ipdb.set_trace()
        s = image.get_shape().as_list()
        assert len(s) == 4, "Image needs to be of shape [batch, width, height, channel]"
        scale = tf.reshape(scale, [-1])
        crop_location = tf.cast(crop_location, tf.float32)
        crop_location = tf.reshape(crop_location, [s[0], 2])
        crop_size = tf.cast(crop_size, tf.float32)

        crop_size_scaled = crop_size / scale
        y1 = crop_location[:, 0] - crop_size_scaled//2
        y2 = y1 + crop_size_scaled
        x1 = crop_location[:, 1] - crop_size_scaled//2
        x2 = x1 + crop_size_scaled
        y1 /= s[1]
        y2 /= s[1]
        x1 /= s[2]
        x2 /= s[2]
        boxes = tf.stack([y1, x1, y2, x2], -1)

        crop_size = tf.cast(tf.stack([crop_size, crop_size]), tf.int32)
        box_ind = tf.range(s[0])
        image_c = tf.image.crop_and_resize(tf.cast(image, tf.float32), boxes, box_ind, crop_size, name='crop')
        return image_c

    
def create_multiple_gaussian_map(coords_uv, output_size, sigma, valid_vec=None):
    """ 
    Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
    with variance sigma for multiple coordinates.
    """
    with tf.name_scope('create_multiple_gaussian_map'):
        sigma = tf.cast(sigma, tf.float32)
        assert len(output_size) == 2
        s = coords_uv.get_shape().as_list()
        coords_uv = tf.cast(coords_uv, tf.int32)
        if valid_vec is not None:
            valid_vec = tf.cast(valid_vec, tf.float32)
            valid_vec = tf.squeeze(valid_vec)
            cond_val = tf.greater(valid_vec, 0.5)
        else:
            cond_val = tf.ones_like(coords_uv[:, 0], dtype=tf.float32)
            cond_val = tf.greater(cond_val, 0.5)

        cond_1_in = tf.logical_and(tf.less(coords_uv[:, 0], output_size[0]-1), tf.greater(coords_uv[:, 0], 0))
        cond_2_in = tf.logical_and(tf.less(coords_uv[:, 1], output_size[1]-1), tf.greater(coords_uv[:, 1], 0))
        cond_in = tf.logical_and(cond_1_in, cond_2_in)
        cond = tf.logical_and(cond_val, cond_in)

        coords_uv = tf.cast(coords_uv, tf.float32)

        # create meshgrid
        x_range = tf.expand_dims(tf.range(output_size[0]), 1)
        y_range = tf.expand_dims(tf.range(output_size[1]), 0)

        X = tf.cast(tf.tile(x_range, [1, output_size[1]]), tf.float32)
        Y = tf.cast(tf.tile(y_range, [output_size[0], 1]), tf.float32)

        X.set_shape((output_size[0], output_size[1]))
        Y.set_shape((output_size[0], output_size[1]))

        X = tf.expand_dims(X, -1)
        Y = tf.expand_dims(Y, -1)

        X_b = tf.tile(X, [1, 1, s[0]])
        Y_b = tf.tile(Y, [1, 1, s[0]])

        X_b -= coords_uv[:, 0]
        Y_b -= coords_uv[:, 1]

        dist = tf.square(X_b) + tf.square(Y_b)

        scoremap = tf.exp(-dist / tf.square(sigma)) * tf.cast(cond, tf.float32)

        return scoremap


def test1():
    #ipdb.set_trace()
    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    data_path = "../data/"
    #data_path = "E:/data/"
    tfrecords_path = data_path+"TFRecords/evaluation_of_RHD.tfrecords"
    
    ## sess config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    ## dataset para
    batch_size = 8
    n_examples = 2728
    max_iter = 1000
    
    n_epochs = int(batch_size*max_iter/n_examples) + 1
    
    dataset = create_dataset_from_TFRecords(tfrecords_path, batch_size=batch_size, is_shuffle=True, n_epochs=n_epochs)
    
    # create iterator
    iterator = dataset.make_initializable_iterator()
    sess.run(iterator.initializer) 

    img_crop, scoremap, keypoint_vis21 = iterator.get_next()


    for _ in range(max_iter):
        #ipdb.set_trace()
        img_crop_v, scoremap_v, keypoint_vis21_v = sess.run([img_crop, scoremap, keypoint_vis21])
        print(img_crop_v.shape, scoremap_v.shape, keypoint_vis21_v.shape)
    
    print("TEST DONE!")
    

def test2():
    #ipdb.set_trace()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    data_path = "../data/"
    train_tfrecords_path = data_path+"TFRecords/test_of_RHD_test.tfrecords"
    
    ## dataset para
    batch_size = 8
    n_examples = 2728
    max_iter = 1000
    
    n_epochs = int(batch_size*max_iter/n_examples) + 1

    train_dataset = create_dataset_from_TFRecords(train_tfrecords_path, batch_size=batch_size, is_shuffle=True, n_epochs=n_epochs)
    
    # 创建一个feedable iterator
    handle = tf.placeholder(tf.string, [])
    feed_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    img_crop, scoremap = feed_iterator.get_next()

    # 创建不同的iterator
    train_iterator = train_dataset.make_one_shot_iterator()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    

    # 生成对应的handle
    train_handle = sess.run(train_iterator.string_handle())

    for _ in range(max_iter):
        ipdb.set_trace()
        print(sess.run([img_crop, scoremap], feed_dict={handle: train_handle}))
            
                
    
    print("TEST DONE!")
    

if __name__ == '__main__':
    fire.Fire(test1)
