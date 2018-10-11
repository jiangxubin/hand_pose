#coding=utf-8

'''
@author LiangYu
@email  liangyufz@gmail.com
@create date 2018-07-12 10:57:22
@modify date 2018-08-15 10:57:22
@desc [description]
'''
class Config(object):
    """ 
    Recommend change params from the command line : 
    (every time)train_mark, dataset_type
    (optionally)with_eval, batch_size, max_iter, use_gpu, gpu_id, etc.
    """
    
    config_name = "Configs of Hand Pose Estimation"

    # data
    data_path = "../data/"
    dataset_type = ""  # op
    tfrecords_path_dict = {
        "RHD": ["TFRecords/train_of_RHD_41258.tfrecords", "TFRecords/eval_of_RHD_2728.tfrecords"],
        "RHD_slim": ["TFRecords/train_of_RHD_slim_38503.tfrecords", "TFRecords/eval_of_RHD_slim_2617.tfrecords"],
        "STB": ["TFRecords/train_of_STB_15000.tfrecords", "TFRecords/eval_of_STB_3000.tfrecords"],
        "STB_320": ["TFRecords/train_of_STB_320_15000.tfrecords", "TFRecords/eval_of_STB_320_3000.tfrecords"],
        "RS_320": ["TFRecords/train_of_RHD_STB_320_56258.tfrecords", "TFRecords/eval_of_RHD_STB_320_5728.tfrecords"]
    }

    #### dataloader param  
    crop_size = 256
    image_size_h = 320#480
    image_size_w = 320#640
    sigma = 25.0
    box_scale = 1.2
    train_vis = True
    augument = False
    coord_uv_noise_sigma = 2.5  # std dev in px of noise on the uv coordinates
    crop_center_noise_sigma = 20.0  # std dev in px: this moves what is in the "center", but the crop always contains all keypoints
    need_heatmap_gt = True
    ####

    # train params
    train_mark = "123456 etc"
    batch_size = 8
    lr = [1e-4, 1e-5, 1e-6]
    lr_iter = [10000, 20000]
    max_iter = 30000
    with_eval = False
    with_dsnt = False

    # log & save params        
    show_loss_freq = 1000
    snapshot_freq = 5000
    snapshot_path = "../data/snapshots/"
    logdir = "./logs/"
    
    
    # eval params
    plot_eval = False
    save_eval_plot_path = data_path + 'eval_result/' 
    save_result_path = "./result/"
    eval_vis = "all"
    load_path = "OriginPaper" #"0714_posenet_n3_R_c1.2_b8_i50000"


    # test params
    test_data_path = 'hand_pose_test'
    save_test_plot_path = data_path + 'test_result/' 


    # general reference params
    timeformat1 = "%m%d_"
    timeformat2 = "[%m%d_%H%M]"     

    # GPU params
    use_gpu = True #if "Linux" in platform.system() else False
    gpu_id = "0"



opt = Config()
    
