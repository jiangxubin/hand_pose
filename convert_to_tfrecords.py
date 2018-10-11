#coding=utf-8

'''
@author LiangYu
@email  liangyufz@gmail.com
@create date 2018-07-19 13:56:11
@modify date 2018-08-15 11:05:14
@desc [description]
'''

import os
import time
import random
import fire
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2 as cv
from utils.plot_utlis import plot_hand
from utils.data_utils import load_json, store_json, convert_kp, _bytes_feature

#--data_path:
#    ├── RHD(RHD_published_v2)
#    ├── STB(STB)    
#    ├── MUL(multiview_hand_pose_dataset)
#    ├── CMU(hand_labels, hand_labels_synth, hand143_panopticdb)
#    └── TFRecords
#
#FOR EXAMPLE:
#--STB:
#    ├── image_names.json(train, eval, all): load_json(data_path + dataset_type + image_names.json)
#    └── data_folders(a.jpg/png & a.json): image and annotation have the same name
#
#--annotation.json: {
#            "image_name" : a, # string
#            "hand_joints" : [[u, v, visible], [...], ...] # list of 21 keypoints, shape (21,3)
#            }


data_path = "../data/"

dataset_path_dict = {
    "RHD": "RHD_published_v2",
    "STB": "STB",
    "STB_320": "STB_320"
}

dataset_json_dict = {
    "RHD": ["rhd_img_names_train_41258.json", "rhd_img_names_eval_2728.json"],
    "STB": ["SK_img_names_train_15000.json", "SK_img_names_eval_3000.json", "SK_img_names_all_18000.json"],
    "STB_320": ["SK_img_names_train_15000.json", "SK_img_names_eval_3000.json", "SK_img_names_all_18000.json"]
}

def convert_to_TFRecords(dataset_list, state="train"):
    """Convert dataset in the list to TFRecords"""
    dataset_list = dataset_list.split("/")
    tfrecords_filename = data_path + "TFRecords/" + state + "_of_" + "_".join(dataset_list) + "_num.tfrecords"
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    print("Writing into ", tfrecords_filename)

    samples = 0
    begin = time.time()
    for _, dataset_type in enumerate(dataset_list):
        mid = time.time()
        print("Process dataset:", dataset_type)
        dataset_path = data_path + dataset_path_dict[dataset_type]
        image_names_json = dataset_json_dict[dataset_type][0] if state == "train" else dataset_json_dict[dataset_type][1]
        image_names = load_json(dataset_path + "/" + image_names_json)
        image_type = ".png" if dataset_type in ["RHD", "STB", "STB_320"] else ".jpg"
        for _, image_name in enumerate(image_names):
            write_one_example(dataset_path, image_name, image_type, writer)
            samples += 1 
        print("Cost time:", time.time() - mid)

    writer.close()
    os.rename(tfrecords_filename, tfrecords_filename.replace("num", str(samples)))        
    print("ALL Done! Cost total time:", time.time() - begin)

def write_one_example(dataset_path, image_name, image_type, writer):
    img_raw = cv.imread(dataset_path + "/" + image_name + image_type)
    img_raw = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)
    anno_infos = load_json(dataset_path + "/" + image_name + ".json")   
    
    name = bytes(anno_infos['img_name'], encoding='utf-8')
    joints = np.array(anno_infos['hand_pts']).astype('float32')
    
    #bytes/int64/float 写入
    image_bytes = img_raw.tostring()
    joints_bytes = joints.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        "image": _bytes_feature(image_bytes),                    
        "name": _bytes_feature(name),
        "joints": _bytes_feature(joints_bytes)
    }))
    writer.write(example.SerializeToString())  

def read_STB(cam="SK", saveinfo=True, vis=False):
    """read the STB xyz 3D annotation, calculate the 2D annotation and store in json"""
    dataset_path = data_path + dataset_path_dict["STB"]
    sequences = ["B1Counting", "B1Random", "B2Counting", "B2Random", "B3Counting", "B3Random", "B4Counting", "B4Random", "B5Counting", "B5Random", "B6Counting", "B6Random"]
    img_names_all = list()
    eval_img_names_all = list()
    eval_num_seq = 250

    if cam == 'BB':
        fx = 822.79041
        fy = 822.79041
        tx = 318.47345
        ty = 250.31296
        base = 120.054

        K = np.diag([fx, fy, 1.0])
        K[0, 2] = tx
        K[1, 2] = ty
        M1 = np.column_stack((K, np.zeros((3, 1))))

        M2_l = np.eye(4)
        M2_r = M2_l.copy()
        M2_r[0, 3] = -base
    else:
        fx_color = 607.92271
        fy_color = 607.88192
        tx_color = 314.78337
        ty_color = 236.42484
        
        fx_depth = 475.62768
        fy_depth = 474.77709
        tx_depth = 336.41179
        ty_depth = 238.77962

        K_color = np.diag([fx_color, fy_color, 1.0])
        K_color[0, 2] = tx_color
        K_color[1, 2] = ty_color
        #M1_color = np.column_stack((K_color, np.zeros((3, 1))))

        K_depth = np.diag([fx_depth, fy_depth, 1.0])
        K_depth[0, 2] = tx_depth
        K_depth[1, 2] = ty_depth
        #M1_depth = np.column_stack((K_depth, np.zeros((3, 1))))

        rvec = np.array([0.00531, -0.01196, 0.00301])
        tvec = np.array([-24.0381, -0.4563, -1.2326])
    
    print("saveinfo:", saveinfo, "vis", vis)
    for _, sequence in enumerate(sequences):
        print("Working on ", sequence)
        mat_path = dataset_path + "/labels/" + sequence + "_" + cam + ".mat"
        load_mat = sio.loadmat(mat_path)
        handPara = load_mat["handPara"]
        img_names_seq = list()

        for i in range(1500):
            if cam == "BB":
                img_left_name = sequence + "/" + cam + "_left_" + str(i)
                img_right_name = sequence + "/" + cam + "_right_" + str(i)
                img_names_seq.append(img_left_name)
                img_names_seq.append(img_right_name)

                img_left = dataset_path + "/" + img_left_name + ".png"
                img_right = dataset_path + "/" + img_right_name + ".png"
                
                anno_xyz = np.row_stack((handPara[:, :, i], np.ones((1, 21))))
                
                M_l = np.dot(M1, M2_l)
                anno_uv_l = np.dot(M_l, anno_xyz) 
                anno_uv_l = anno_uv_l / anno_uv_l[2, :]

                M_r = np.dot(M1, M2_r)
                anno_uv_r = np.dot(M_r, anno_xyz) 
                anno_uv_r = anno_uv_r / anno_uv_r[2, :]
                
                joints_uv_l = convert_kp(np.transpose(anno_uv_l, [1, 0]))
                joints_uv_r = convert_kp(np.transpose(anno_uv_r, [1, 0]))
                
                if saveinfo:
                    anno_infos_left = {}
                    anno_infos_left['img_name'] = img_left_name
                    anno_infos_left['hand_pts'] = joints_uv_l.tolist()
                    store_json(img_left.replace(".png", ".json"), anno_infos_left)
                    anno_infos_right = {}
                    anno_infos_right['img_name'] = img_right_name
                    anno_infos_right['hand_pts'] = joints_uv_r.tolist()
                    store_json(img_right.replace(".png", ".json"), anno_infos_right)

                if vis: # vis the keypoints and save        
                    img_left_raw, img_right_raw = cv.imread(img_left), cv.imread(img_right)
                    img_left_raw = cv.cvtColor(img_left_raw, cv.COLOR_BGR2RGB)
                    img_right_raw = cv.cvtColor(img_right_raw, cv.COLOR_BGR2RGB)
                    joints_hw_l, joints_hw_r = joints_uv_l[:, ::-1][:, 1:], joints_uv_r[:, ::-1][:, 1:]
        
                    fig = plt.figure(1, figsize=(10, 8))
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)
                    ax1.set_title(cam + "_left_" + str(i))
                    ax1.imshow(img_left_raw)
                    ax2.set_title(cam + "_right_" + str(i))
                    ax2.imshow(img_right_raw)
                    plot_hand(joints_hw_l, ax1)
                    plot_hand(joints_hw_r, ax2)
                    savedir = dataset_path + "/" + sequence + "_vis"    
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    plt.savefig(savedir + "/" + cam + "_left_right_" + str(i) + ".png")
                    plt.close('all')
            elif cam == "useless": ## useless  for test
                img_color = dataset_path + "/" + sequence + "/" + cam + "_color_" + str(i) + ".png"
                anno_xyz = handPara[:, :, i]
                #anno_xyz = np.row_stack((handPara[:, :, i], np.ones((1, 21))))
                joints_uv_depth_raw = np.dot(K_depth, anno_xyz)
                
                # P2 = R21(P1-T1) + T2
                R12, _ = cv.Rodrigues(rvec)
                
                R21 = np.mat(R12).I
                
                #a = np.dot(R12, R21)
                a = np.array(R21.getA())
                joints_uv_color_raw = np.dot(a, joints_uv_depth_raw)
                #b = np.transpose(joints_uv_color_raw, [1,0])
                #c = b - tvec
                #joints_uv_color_raw = np.dot(R21, joints_uv_depth_raw) + tvec
                joints_uv_color_raw = joints_uv_color_raw / joints_uv_color_raw[2, :]

                joints_uv_color = convert_kp((np.transpose(joints_uv_color_raw, [1, 0]) - tvec)[:, :2])

                if vis:
                    img_color_raw = cv.imread(img_color)
                    img_color_raw = cv.cvtColor(img_color_raw, cv.COLOR_RGB2BGR)
                    joints_hw_color = joints_uv_color[:, ::-1]

                    fig = plt.figure(1, figsize=(10, 8))
                    ax1 = fig.add_subplot(111)
                    ax1.set_title(cam + "_color_" + str(i)) 
                    ax1.imshow(img_color_raw)
                    plot_hand(joints_hw_color, ax1)
            else: # SK
                img_color_name = sequence + "/" + cam + "_color_" + str(i)
                img_names_seq.append(img_color_name)

                img_color = dataset_path + "/" + img_color_name + ".png"
                anno_xyz = handPara[:, :, i]   
                anno_xyz = np.transpose(anno_xyz, [1, 0])
                joints_uv_color_raw, _ = cv.projectPoints(anno_xyz, rvec*(-1), tvec*(-1), K_color, np.zeros((5, 1)))
                
                joints_uv_color = convert_kp(joints_uv_color_raw[:, 0, :])
                joints_uv_color = np.column_stack((joints_uv_color, np.ones((21, 1))))
                if saveinfo:
                    anno_infos_color = {}
                    anno_infos_color['img_name'] = img_color_name
                    anno_infos_color['hand_pts'] = joints_uv_color.tolist()
                    store_json(img_color.replace(".png", ".json"), anno_infos_color)
                if vis:
                    img_color_raw = cv.imread(img_color)
                    img_color_raw = cv.cvtColor(img_color_raw, cv.COLOR_RGB2BGR)
                    joints_hw_color = joints_uv_color[:, ::-1][:, 1:]

                    fig = plt.figure(1, figsize=(10, 8))
                    ax1 = fig.add_subplot(111)
                    ax1.set_title(cam + "_color_" + str(i)) 
                    ax1.imshow(img_color_raw)
                    plot_hand(joints_hw_color, ax1)
                    savedir = dataset_path + "/" + sequence + "_vis"    
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    plt.savefig(savedir + "/" + cam + "_color_" + str(i) + ".png")
                    plt.close('all')
                #return img_color, pose_2d
            #break
        # end of every seq

        random_name_list = []
        while len(random_name_list) < eval_num_seq:
            random_name = random.choice(img_names_seq)
            if random_name not in random_name_list:
                random_name_list.append(random_name)    
        
        eval_img_names_all.extend(random_name_list)
        img_names_all.extend(img_names_seq)
        #break
    #end of all

    train_img_names_all = [img_name for img_name in img_names_all if img_name not in eval_img_names_all]
    if saveinfo:
        store_json(dataset_path + "/" + cam + "_img_names_all_" + str(len(img_names_all)) + ".json", img_names_all)
        store_json(dataset_path + "/" + cam + "_img_names_train_" + str(len(train_img_names_all)) + ".json", train_img_names_all)
        store_json(dataset_path + "/" + cam + "_img_names_eval_" + str(len(eval_img_names_all)) + ".json", eval_img_names_all)
    else:
        return img_names_all


def crop_to_size(dataset_type="STB", crop_size=320):
    """crop the dataset to crop_size"""
    half_crop_size = crop_size/2
    dataset_path = data_path + dataset_path_dict[dataset_type]
    image_names_json = dataset_json_dict[dataset_type][2]
    image_names = load_json(dataset_path + "/" + image_names_json)
    image_type = ".png" if dataset_type in ["RHD", "STB"] else ".jpg"

    save_dir = dataset_path + "_" + str(crop_size)

    begin = time.time()
    print("crop {}:{} to size {}".format(dataset_type, len(image_names), crop_size))
    for _, image_name in enumerate(image_names):
        img_raw = cv.imread(dataset_path + "/" + image_name + image_type)
        #img_raw = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)
        anno_infos = load_json(dataset_path + "/" + image_name + ".json")
        joints = np.array(anno_infos['hand_pts']).astype("float32")

        img_h, img_w, _ = img_raw.shape
        crop_center = joints[:, :2][12].astype(int)
        half_size = min(crop_center[0], crop_center[1], half_crop_size, img_w - crop_center[0], img_h - crop_center[1])
        
        x0, y0 = (crop_center - half_size).astype(int)
        x1, y1 = (crop_center + half_size).astype(int)
        img_crop = img_raw[y0:y1, x0:x1]
        joints[:, :2] -= [x0, y0]

        scale = half_crop_size/half_size
        if scale > 1:
            img_crop = cv.resize(img_crop, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
            joints[:, :2] *= scale

        if not os.path.exists(save_dir + "/" + image_name.split("/")[0]):
            print("Make dirs:", save_dir + "/" + image_name.split("/")[0])
            os.makedirs(save_dir + "/" + image_name.split("/")[0])        
        cv.imwrite(save_dir + "/" + image_name + image_type, img_crop)
        anno_infos = {}
        anno_infos['img_name'] = image_name
        anno_infos['hand_pts'] = joints.tolist()
        store_json(save_dir + "/" + image_name + ".json", anno_infos)

    print("Done! Cost time:", time.time() - begin)
    store_json(save_dir + "/" + image_names_json, image_names)


if __name__ == '__main__':
    fire.Fire()
