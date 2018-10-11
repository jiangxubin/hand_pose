#coding=utf-8

'''
@author LiangYu
@email  liangyufz@gmail.com
@create date 2018-07-12 14:20:20
@modify date 2018-08-15 10:56:18
@desc [description]
'''

import sys
import os
import time
import csv
import platform
import warnings
import fire
import ipdb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from config import opt
from models.PoseNet import PoseNet
from models.dsnt import js_reg_loss
from data.data_loader import create_dataset_from_TFRecords
from utils.model_utils import LearningRateScheduler, load_weights_from_snapshot
from utils.eval_utils import EvalUtil, detect_keypoints
from utils.plot_utlis import plot_hand, plot_prob

# global param


warnings.filterwarnings("ignore")

def parse(kwargs):
    # parse config and args
    for k, v in kwargs.items():
        if not hasattr(opt, k):
            print("Warning: opt has not attribute %s" % k)
        setattr(opt, k, v)
        print("set:--{}: {}".format(k, v))

def train(**kwargs):
    #ipdb.set_trace()
    parse(kwargs)
    
    # sess config
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))   
    sess_config = tf.ConfigProto()
    if opt.use_gpu:
        #print(time.strftime(opt.timeformat2) + 'GPU for training, and GPU id is ', opt.gpu_id)
        #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, opt.gpu_id)))  # (0, 1) -> ['0', '1'] -> "0,1"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
        sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    
    # snapshot dir
    dir_name = time.strftime(opt.timeformat1) + "n{opt.train_mark}_{opt.dataset_type}_c{opt.box_scale}_b{opt.batch_size}_a{opt.augument}_v{opt.train_vis}_d{opt.with_dsnt}_i{a}w".format(opt=opt, a=opt.max_iter/10000)
    #setattr(opt, "load_path", dir_name)
    snapshot_path = opt.snapshot_path + dir_name
    setattr(opt, "snapshot_path", snapshot_path)
    if not os.path.exists(opt.snapshot_path):
        os.makedirs(opt.snapshot_path)
        print(time.strftime(opt.timeformat2) + 'Created snapshot dir:', opt.snapshot_path)
    else:
        print(time.strftime(opt.timeformat2) + 'snapshot dir exist!', opt.snapshot_path)
    
    # show train params
    print(time.strftime(opt.timeformat2) + "dataset_type:({opt.dataset_type}) batch_size:({opt.batch_size}) max_iter:({opt.max_iter})".format(opt=opt))    

    # create dataset
    n_train_examples = int(opt.tfrecords_path_dict[opt.dataset_type][0].split(".")[0].split("_")[-1])
    n_epochs = int(opt.batch_size*opt.max_iter/n_train_examples) + 1
    print(time.strftime(opt.timeformat2) + "n_train_examples:{}  n_epochs:{}".format(n_train_examples, n_epochs))
    train_dataset = create_dataset_from_TFRecords(opt.data_path + opt.tfrecords_path_dict[opt.dataset_type][0], state="train", batch_size=opt.batch_size, is_shuffle=True, n_epochs=n_epochs)

    if opt.with_eval:
        # create eval dataset
        n_eval_examples = int(opt.tfrecords_path_dict[opt.dataset_type][1].split(".")[0].split("_")[-1])
        print(time.strftime(opt.timeformat2) + "n_eval_examples:{}".format(n_eval_examples))
        eval_dataset = create_dataset_from_TFRecords(opt.data_path + opt.tfrecords_path_dict[opt.dataset_type][1], state="eval", batch_size=1, is_shuffle=False, n_epochs=1)
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        img_crop, scoremap, keypoint_vis21, keypoint_uv21, scale, name = iterator.get_next()
        train_init_op = iterator.make_initializer(train_dataset)
        eval_init_op = iterator.make_initializer(eval_dataset)
        sess.run(train_init_op)
    else:    
        # create iterator
        train_iterator = train_dataset.make_initializable_iterator()
        sess.run(train_iterator.initializer)

        img_crop, scoremap, keypoint_vis21, keypoint_uv21, scale, name = train_iterator.get_next()
    

    if opt.with_dsnt:
        # build network
        net = PoseNet()
        norm_heatmap, coords_pred, scoremap_list = net.inference_with_dsnt(img_crop, train=True)
        coords_pred = (coords_pred + 1)/2
        tf.summary.histogram("norm_heatmap", norm_heatmap)
        tf.summary.histogram("coords_pred", coords_pred)
        tf.summary.histogram("scoremap_list", scoremap_list)

        # loss
        loss = 0.0
        coords_gt_unlimit = tf.reshape(keypoint_uv21, (-1, 2)) / opt.crop_size
        
        loss_1 = tf.losses.mean_squared_error(coords_gt_unlimit, coords_pred)
        loss_11 = tf.reduce_mean(tf.square(coords_gt_unlimit - coords_pred))
        
        coords_gt = tf.clip_by_value(coords_gt_unlimit, 0.0, 1.0)
        loss_2 = js_reg_loss(norm_heatmap, coords_gt) 
        loss = loss_1 + loss_2
        tf.summary.histogram("coords_gt", coords_gt)
        tf.summary.histogram("coords_gt_unlimit", coords_gt_unlimit)
        tf.summary.histogram("loss_1", loss_1)
        tf.summary.histogram("loss_11", loss_11)
        tf.summary.histogram("loss_2", loss_2)

    else:
        # build network
        net = PoseNet() 
        keypoints_scoremap, heatmap_dice, _ = net.inference(img_crop, train=True)
        shape = scoremap.shape.as_list()
        keypoints_scoremap = [tf.image.resize_images(x, (shape[1], shape[2])) for x in keypoints_scoremap]
        #temp_scoremap = [tf.image.resize_images(x, (shape[1], shape[2])) for x in temp_scoremap]
        #tf.summary.histogram('heatmap_dice', heatmap_dice)
        tf.summary.histogram('scoremap_pred', keypoints_scoremap)
        #tf.summary.histogram('scoremap_pred_temp', temp_scoremap)

        # Loss
        loss = 0.0
        vis = tf.cast(tf.reshape(keypoint_vis21, [opt.batch_size, shape[3]]), tf.float32)
        vis_sum = tf.reduce_sum(vis)
        #for i, pred_item in enumerate(keypoints_scoremap):
        for i, pred_item in enumerate(keypoints_scoremap):
            loss_mid = tf.sqrt(tf.reduce_mean(tf.square(pred_item - scoremap), [1, 2]))
            #loss += tf.reduce_sum(loss_mid) / (opt.batch_size*21)
            if opt.train_vis:
                loss += tf.reduce_mean(loss_mid)
            else:
                loss += tf.reduce_sum(vis*loss_mid)/(vis_sum + 0.0001)

    
    # Solver
    global_step = tf.Variable(0, trainable=False, name="global_step")
    lr_scheduler = LearningRateScheduler(values=opt.lr, steps=opt.lr_iter)
    lr = lr_scheduler.get_lr(global_step)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss)

    # log 
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('scoremap_gt', scoremap)
    tf.summary.histogram("scale", scale)
    #tf.summary.text("name", name)
    #tf.summary.image("train_image", img_crop)
    #tf.summary.image("scoremap", scoremap[-1][:, :, 0])
    merged_summary_op = tf.summary.merge_all() 
    
    # init 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=2.0)
    summary_writer = tf.summary.FileWriter(logdir=opt.logdir + dir_name, graph=sess.graph)


    # initialize network weights
    if opt.load_path != "OriginPaper":
        # retrained version
        load_path = opt.snapshot_path.replace(dir_name, opt.load_path)
        last_cpt = tf.train.latest_checkpoint(load_path)
        assert last_cpt is not None, time.strftime(opt.timeformat2) + "Could not locate snapshot to load"
        print(time.strftime(opt.timeformat2) + "load_weights_from_snapshot:" + load_path)
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    else:
        rename_dict = {'CPM/PoseNet': 'PoseNet2D', '_CPM': ''}
        load_weights_from_snapshot(sess, './weights/cpm-model-mpii', ['PersonNet', 'PoseNet/Mconv', 'conv5_2_CPM'], rename_dict)        

    #ipdb.set_trace()
    print(time.strftime(opt.timeformat2) + 'Starting to train ...')
    begin = time.time()
    cost_time_list = list()
    loss_list = list()
    # count = 0
    try:
        for i in range(opt.max_iter):
            #_, loss_v1, loss_v2, summary, keypoints_scoremap_v, heatmap_dice_v, temp_scoremap_v, scoremap_v = sess.run([train_op, loss_1, loss_2, merged_summary_op, keypoints_scoremap, heatmap_dice, temp_scoremap, scoremap])
            _, loss_v, summary = sess.run([train_op, loss, merged_summary_op])
            #coords_gt_v, name_v = sess.run([coords_gt, name])

            #if coords_gt_v.max() > 1.0 or coords_gt_v.min() < 0:
            #    print(str(name_v[0]).split("'")[-2])
            #else:
            #    count += 1
            #    continue

            if (i % opt.show_loss_freq) == 0:
                loss_list.append(loss_v)
                cost_time_per_freq = time.time() - begin
                cost_time_list.append(cost_time_per_freq)
                print(time.strftime(opt.timeformat2) + 'Iteration %d\t Loss %.5f' % (i, loss_v), 'Cost_time:', cost_time_per_freq)
                sys.stdout.flush()
            
            if (i % opt.snapshot_freq) == 0:
                saver.save(sess, "%s/model" % opt.snapshot_path, global_step=i)
                print(time.strftime(opt.timeformat2) + 'Saved a snapshot.')

                if opt.with_eval:
                    sess.run(eval_init_op)
                    print(time.strftime(opt.timeformat2) + "Evaluate the " + dir_name)
                    auc11, auc01 = eval_in_sess(sess, n_eval_examples, keypoints_scoremap, img_crop, keypoint_uv21, keypoint_vis21, scale, name)
                    print(time.strftime(opt.timeformat2) + 'Evaluation results: AUC: {a:.3f}({b:.3f})'.format(a=auc11, b=auc01))
                    sess.run(train_init_op)
                sys.stdout.flush()

            summary_writer.add_summary(summary, global_step=i)

    except tf.errors.OutOfRangeError:
        print(time.strftime(opt.timeformat2) + "out of range error")
    
    summary_writer.close()
    cost_time = time.time() - begin
    cost_time_list.append(cost_time)
    print(time.strftime(opt.timeformat2) + 'Training finished. Saving final snapshot.')
    print(time.strftime(opt.timeformat2) + 'Cost time:', cost_time)
    saver.save(sess, "%s/model" % opt.snapshot_path, global_step=opt.max_iter)
    sess.close()
    setattr(opt, "load_path", dir_name)
    save_train_infos(cost_time_list, loss_list)
    
    return dir_name
    
    
def eval_in_sess(sess, n_eval_examples, keypoints_scoremap, img_crop, keypoint_uv21, keypoint_vis21, scale, name):
    
    util11 = EvalUtil(vis="11")
    util01 = EvalUtil(vis="01")

    begin = time.time()
    names_list = list()
    try:
        for _ in range(n_eval_examples):   
            keypoints_scoremap_v, img_crop_v, kp_uv21_gt_v, kp_vis_v, scale_v, name_v = sess.run([keypoints_scoremap, img_crop, keypoint_uv21, keypoint_vis21, scale, name])
            keypoints_scoremap_v = keypoints_scoremap_v[-1] 
            name_v = str(name_v[0]).split("'")[-2]
            if name_v not in names_list:
                names_list.append(name_v)

            img_crop_v = img_crop_v[0] + 0.5
            keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)  #256*256*21
            kp_uv21_gt = np.squeeze(kp_uv21_gt_v) #21*2
            kp_vis = np.squeeze(kp_vis_v) #21
            crop_scale = np.squeeze(scale_v)
            # detect keypoints
            coord_hw_pred_crop, _ = detect_keypoints(np.squeeze(keypoints_scoremap_v)) #21*2
            coord_uv_pred_crop = np.stack([coord_hw_pred_crop[:, 1], coord_hw_pred_crop[:, 0]], 1) #21*2

            util11.feed(kp_uv21_gt/crop_scale, kp_vis, coord_uv_pred_crop/crop_scale)
            util01.feed(kp_uv21_gt/crop_scale, kp_vis, coord_uv_pred_crop/crop_scale)

    except tf.errors.OutOfRangeError:
        print(time.strftime(opt.timeformat2) + "out of range error")
    
    cost_time = time.time() - begin

    eval_dict11 = util11.get_measures(0.0, 30.0, 20)
    eval_dict01 = util01.get_measures(0.0, 30.0, 20)

    print(time.strftime(opt.timeformat2) + "Eval Done! Process {} images, Cost time:{}".format(len(names_list), cost_time))
    
    return eval_dict11["auc"], eval_dict01['auc']

def save_train_infos(cost_time_list, loss_list):
    infos = []
    if not os.path.exists(opt.save_result_path + "train_infos.csv"):
        title = ["Snapshots", "Cost_time", "final_loss", "", ""]
        infos.append(title)
        
    train_infos = [opt.load_path, cost_time_list[-1], loss_list[-1], "||cost_time:"]
    train_infos.extend(cost_time_list)
    train_infos.append("||loss:")
    train_infos.extend(loss_list)
    infos.append(train_infos)

    with open(opt.save_result_path + "train_infos.csv", "a", newline='') as csvfile: 
        writer = csv.writer(csvfile)
        for _, info in enumerate(infos):
            writer.writerow(info)

    
    
def evaluation(**kwargs):
    #ipdb.set_trace()
    parse(kwargs)
    #for _, (attribute, value) in enumerate(opt.__class__.__dict__.items()): #  iter __dict__ attr
    #for _, attribute in enumerate(dir(opt)): # iter all attr
    #    if not attribute.startswith("__"):
    #    if attribute in ['data_path', 'batch_size']:
    #        print(attribute, value)
    

    # sess config
    sess_config = tf.ConfigProto()
    if opt.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
        sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config) 
    
    # plot save path
    dir_name = "RESULT_OF_" + opt.load_path
    if opt.plot_eval:
        save_eval_plot_path = opt.save_eval_plot_path + dir_name
        setattr(opt, "save_eval_plot_path", save_eval_plot_path)    
        if not os.path.exists(opt.save_eval_plot_path):
            os.makedirs(opt.save_eval_plot_path)
            print(time.strftime(opt.timeformat2) + 'Created save_eval_plot_path:', opt.save_eval_plot_path)
        else:
            print(time.strftime(opt.timeformat2) + 'save_eval_plot_path exist!', opt.save_eval_plot_path)
    
    # show eval params
    print(time.strftime(opt.timeformat2) + "dataset_type:({opt.dataset_type})  load_path:({opt.load_path}) plot_eval:({opt.plot_eval}) train_vis:({opt.train_vis})".format(opt=opt))  
    
    # create dataset
    n_eval_examples = int(opt.tfrecords_path_dict[opt.dataset_type][1].split(".")[0].split("_")[-1])
    eval_dataset = create_dataset_from_TFRecords(opt.data_path + opt.tfrecords_path_dict[opt.dataset_type][1], state="eval", batch_size=1, is_shuffle=False, n_epochs=1)

    # create iterator
    eval_iterator = eval_dataset.make_initializable_iterator()
    sess.run(eval_iterator.initializer)

    img_crop, scoremap, keypoint_vis21, keypoint_uv21, scale, name = eval_iterator.get_next()

    if opt.with_dsnt:
        net = PoseNet()
        norm_heatmap, coords_pred, scoremap_list = net.inference_with_dsnt(img_crop, train=True)
        coords_pred = (coords_pred + 1)/2 * opt.crop_size
    else:
        # build network
        net = PoseNet() 
        keypoints_scoremap, _, _ = net.inference(img_crop)
        keypoints_scoremap = keypoints_scoremap[-1]
        shape = scoremap.get_shape().as_list()
        keypoints_scoremap = tf.image.resize_images(keypoints_scoremap, (shape[1], shape[2]))
    
    # initialize network weights
    if opt.load_path != "OriginPaper":
        # retrained version
        load_path = opt.snapshot_path + opt.load_path
        last_cpt = tf.train.latest_checkpoint(load_path)
        assert last_cpt is not None, time.strftime(opt.timeformat2) + "Could not locate snapshot to load"
        print(time.strftime(opt.timeformat2) + "load_weights_from_snapshot:" + load_path)
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    else:
        # load weights used in the paper
        net.init(sess, weight_files=['./weights/posenet-rhd-stb.pickle'], exclude_var_list=['PosePrior', 'ViewpointNet'])
    
    util11 = EvalUtil(vis="11")
    util01 = EvalUtil(vis="01")

    print(time.strftime(opt.timeformat2) + 'Starting to eval ...')
    begin = time.time()
    names_list = list()

    try:
        for i in range(n_eval_examples):   
            #ipdb.set_trace()
            if opt.with_dsnt:
                coords_pred_v, img_crop_v, kp_uv21_gt_v, kp_vis_v, scale_v, name_v = sess.run([coords_pred, img_crop, keypoint_uv21, keypoint_vis21, scale, name])
                coord_uv_pred_crop = np.squeeze(coords_pred_v)
                coord_hw_pred_crop = np.stack([coord_uv_pred_crop[:, 1], coord_uv_pred_crop[:, 0]], 1) #21*2 just for plot              
            else:
                keypoints_scoremap_v, img_crop_v, kp_uv21_gt_v, kp_vis_v, scale_v, name_v = sess.run([keypoints_scoremap, img_crop, keypoint_uv21, keypoint_vis21, scale, name])
                keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)  #256*256*21
                # detect keypoints
                coord_hw_pred_crop, prob = detect_keypoints(np.squeeze(keypoints_scoremap_v)) #21*2
                coord_uv_pred_crop = np.stack([coord_hw_pred_crop[:, 1], coord_hw_pred_crop[:, 0]], 1) #21*2               
                
                
            name_v = str(name_v[0]).split("'")[-2]
            if name_v not in names_list:
                names_list.append(name_v)

            img_crop_v = img_crop_v[0] + 0.5
            kp_uv21_gt = np.squeeze(kp_uv21_gt_v) #21*2
            kp_vis = np.squeeze(kp_vis_v) #21
            crop_scale = np.squeeze(scale_v)
            
            util11.feed(kp_uv21_gt/crop_scale, kp_vis, coord_uv_pred_crop/crop_scale)
            util01.feed(kp_uv21_gt/crop_scale, kp_vis, coord_uv_pred_crop/crop_scale)
            
            if opt.plot_eval:
                fig = plt.figure(1, figsize=(10, 8))
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)
                ax1.set_title("Predicted kp")
                ax1.imshow(img_crop_v)
                plot_hand(coord_hw_pred_crop, ax1)
                ax2.set_title("GroundTruth kp")
                ax2.imshow(img_crop_v)
                coord_hw_gt_crop = np.stack([kp_uv21_gt[:, 1], kp_uv21_gt[:, 0]], 1)
                plot_hand(coord_hw_gt_crop, ax2)
                if not opt.with_dsnt:
                    ax3.set_title("Predicted kp")
                    plot_hand(coord_hw_pred_crop, ax3, prob)
                    ax4.set_title("kp_num—Probability")
                    plot_prob(prob, ax4)
                #plt.show()
                #if not os.path.exists(opt.save_eval_plot_path + "/{}".format(name_v.split("/")[0])) and "/" in name_v :
                #    os.makedirs(opt.save_eval_plot_path + "/{}".format(name_v.split("/")[0]))
                name_v = name_v.replace("/", "_")
                plt.savefig(opt.save_eval_plot_path + "/{}.png".format(name_v))
                plt.close('all')

            if (i % 500) == 0:
                print(time.strftime(opt.timeformat2) + '%d / %d images done: %.3f percent' % (i, n_eval_examples, i*100.0/n_eval_examples))
    except tf.errors.OutOfRangeError:
        print(time.strftime(opt.timeformat2) + "out of range error")

    sess.close()
    
    cost_time = time.time() - begin

    eval_dict11 = util11.get_measures(0.0, 30.0, 20)
    eval_dict01 = util01.get_measures(0.0, 30.0, 20)
    
    #ipdb.set_trace()
    if os.path.exists(opt.save_result_path + dir_name + "_On_" + opt.dataset_type + "_all.npz"):
        print(time.strftime(opt.timeformat2) + "Result file exists")
    else:
        save_result_infos(dir_name, eval_dict11, vis="11", cost_time=cost_time, eval_num=len(names_list))
        if "R" in opt.dataset_type:
            print("a")
            save_result_infos(dir_name, eval_dict01, vis="01", cost_time=cost_time, eval_num=len(names_list))

    print("Eval Done! Process {} images, Cost time:{}".format(len(names_list), cost_time))
    print('Evaluation results:')
    print('Area under curve(all/part): {a:.3f}({b:.3f})'.format(a=eval_dict11['auc'], b=eval_dict01['auc']))
    print('Average mean EPE(all/part): {a:.3f}({b:.3f}) pixels'.format(a=eval_dict11['mean'], b=eval_dict01['mean']))
    print('Average median EPE(all/part): {a:.3f}({b:.3f}) pixels'.format(a=eval_dict11['median'], b=eval_dict01['median']))
    
def save_result_infos(dir_name, eval_dict, vis, cost_time, eval_num):

    vis_type = "all" if vis == "11" else "part"

    mean, median, auc, pck_curve_all, thresholds = eval_dict["mean"], eval_dict['median'], eval_dict['auc'], eval_dict['pck_curve_all'], eval_dict['thresholds']
    
    np.savez(opt.save_result_path + dir_name + "_On_" + opt.dataset_type + "_" + vis_type + ".npz", auc=auc, pck_curve_all=pck_curve_all, thresholds=thresholds)
    
    #ipdb.set_trace()
    
    if "Linux" not in platform.system():
        plt.plot(thresholds, pck_curve_all, linewidth='1', label="{}:{auc:.3f}".format(opt.load_path + "-" + vis_type, auc=auc))
        plt.title("AUC")
        plt.xlabel("Threshold in px")
        plt.ylabel("PCK")
        plt.legend(loc=4)
    
        if "R" in opt.dataset_type and vis_type == "all":
            pass
        else:
            plt.savefig(opt.save_result_path + dir_name + "_On_" + opt.dataset_type + "_" + vis_type + ".png")
            plt.close()


    dir_name = "" if vis == "01" else dir_name
    result_infos = [dir_name, opt.dataset_type, cost_time, eval_num, vis_type, round(auc, 3), round(mean, 3), round(median, 3), "", ""]
    #np.set_printoptions(precision=3)
    result_infos.extend(pck_curve_all.tolist())

    infos = []
    if not os.path.exists(opt.save_result_path + "result_infos.csv"):
        title = ["Snapshots", "eval_dataset", "Cost_time", "Num", "Vis", "AUC", "Mean", "Median", "", "Thresholds"]
        title.extend(thresholds.tolist())
        infos.append(title)
    infos.append(result_infos)
    with open(opt.save_result_path + "result_infos.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for _, info in enumerate(infos):
            writer.writerow(info)    


def test(**kwargs):
    parse(kwargs)

    # sess config
    sess_config = tf.ConfigProto()
    if opt.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
        sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config) 


    # plot save path
    dir_name = "RESULT_OF_" + opt.load_path
    if opt.plot_eval:
        save_test_plot_path = opt.save_test_plot_path + dir_name
        setattr(opt, "save_test_plot_path", save_test_plot_path)    
        if not os.path.exists(opt.save_test_plot_path):
            os.makedirs(opt.save_test_plot_path)
            print(time.strftime(opt.timeformat2) + 'Created save_test_plot_path:', opt.save_test_plot_path)
        else:
            print(time.strftime(opt.timeformat2) + 'save_test_plot_path exist!', opt.save_test_plot_path)


    # network input
    img_crop = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))

    dst_path = opt.data_path + opt.test_data_path   
    image_list = [opt.test_data_path + "/" + img for img in os.listdir(dst_path)]
    
    if opt.with_dsnt:
        net = PoseNet()
        norm_heatmap, coords_pred, scoremap_list = net.inference_with_dsnt(img_crop, train=True)
        coords_pred = (coords_pred + 1)/2 * opt.crop_size
    else:
        # build network
        net = PoseNet() 
        keypoints_scoremap, _, _ = net.inference(img_crop)
        keypoints_scoremap = keypoints_scoremap[-1]
        shape = img_crop.get_shape().as_list()
        keypoints_scoremap = tf.image.resize_images(keypoints_scoremap, (shape[1], shape[2]))
    
    # initialize network weights
    if opt.load_path != "OriginPaper":
        # retrained version
        load_path = opt.snapshot_path + opt.load_path
        last_cpt = tf.train.latest_checkpoint(load_path)
        assert last_cpt is not None, time.strftime(opt.timeformat2) + "Could not locate snapshot to load"
        print(time.strftime(opt.timeformat2) + "load_weights_from_snapshot:" + load_path)
        load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    else:
        # load weights used in the paper
        net.init(sess, weight_files=['./weights/posenet-rhd-stb.pickle'], exclude_var_list=['PosePrior', 'ViewpointNet'])

    begin = time.time()
    for _, name_v in enumerate(image_list):
        #ipdb.set_trace()
        img_crop_v = cv.imread(opt.data_path + "/" + name_v)
        img_crop_v = cv.cvtColor(img_crop_v, cv.COLOR_BGR2RGB)
        img_crop_v = cv.resize(img_crop_v, (256, 256), interpolation=cv.INTER_CUBIC)
        img_crop_tf = np.expand_dims((img_crop_v.astype('float') / 255.0) - 0.5, 0)

        if opt.with_dsnt:
            coords_pred_v = sess.run([coords_pred], feed_dict={img_crop: img_crop_tf})
            coord_uv_pred_crop = np.squeeze(coords_pred_v)
            coord_hw_pred_crop = np.stack([coord_uv_pred_crop[:, 1], coord_uv_pred_crop[:, 0]], 1) #21*2 just for plot              
        else:
            keypoints_scoremap_v = sess.run([keypoints_scoremap], feed_dict={img_crop: img_crop_tf})
            keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)  #256*256*21
            # detect keypoints
            coord_hw_pred_crop, prob = detect_keypoints(np.squeeze(keypoints_scoremap_v)) #21*2
            #coord_uv_pred_crop = np.stack([coord_hw_pred_crop[:, 1], coord_hw_pred_crop[:, 0]], 1) #21*2  


        #keypoints_scoremap_v = sess.run([keypoints_scoremap], feed_dict={img_crop: img_crop_tf})
        #keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)

        #coord_hw_pred_crop, prob = detect_keypoints(np.squeeze(keypoints_scoremap_v))

        # vis
        if opt.plot_eval:

            # plt.imshow(img_crop_v)
            # plot_hand(coord_hw_pred_crop, plt)
        
            fig = plt.figure(1, figsize=(10, 8))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            ax1.set_title("Predicted kp")
            ax1.imshow(img_crop_v)
            plot_hand(coord_hw_pred_crop, ax1)
            ax2.set_title("GroundTruth kp")
            ax2.imshow(img_crop_v)
            #plot_hand(coord_hw_gt_crop, ax2)
            #ax3.set_title("Predicted kp")
            #plot_hand(coord_hw_pred_crop, ax3, prob)
            #ax4.set_title("kp_num—Probability")
            #plot_prob(prob, ax4)
            # plt.show()
            if not os.path.exists(opt.save_test_plot_path + "/{}".format(name_v.split("/")[0])):
                os.makedirs(opt.save_test_plot_path + "/{}".format(name_v.split("/")[0]))
            plt.savefig(opt.save_test_plot_path + "/{}".format(name_v))
            plt.close('all')

    print("TEST DONE! Cost time:", time.time() - begin)


if __name__ == '__main__':
    fire.Fire()
