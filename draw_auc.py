#coding=utf-8

'''
@author LiangYu
@email  liangyufz@gmail.com
@create date 2018-08-06 11:33:34
@modify date 2018-08-06 11:33:34
@desc [description]
'''
import time
import fire
import numpy as np
import matplotlib.pyplot as plt

from config import opt

def parse(kwargs):
    # parse config and args
    for k, v in kwargs.items():
        if not hasattr(opt, k):
            print("Warning: opt has not attribute %s" % k)
        setattr(opt, k, v)
        print("set:--{}: {}".format(k, v))

def draw_single_auc(load_path=opt.load_path, plt=None, label=None, marker="*"):
    np_file = opt.save_result_path + "RESULT_OF_" + load_path + "_On_" + opt.dataset_type + "_" + opt.eval_vis + ".npz"
    data_source = np.load(np_file)
    plt.plot(data_source["thresholds"], data_source["pck_curve_all"], marker=marker, label="{}(auc={auc:.3f})".format(label, auc=data_source["auc"]))
    

def draw_auc(**kwargs):
    parse(kwargs)

    load_path_list = opt.load_path_list.split("/")
    label_list = opt.label_list.split("/")

    marker_list = ["o", "p", "*", "^"]

    plt.title("AUC ON {}".format(opt.dataset_type))
    plt.xlabel("Threshold in px")
    plt.ylabel("PCK")
    for i in range(len(load_path_list)):
        draw_single_auc(load_path=load_path_list[i], plt=plt, label=label_list[i], marker=marker_list[i])
    plt.legend(loc=4)
    plt.savefig(opt.save_result_path + time.strftime(opt.timeformat2) + "AUC_ON_{}.png".format(opt.dataset_type))

if __name__ == '__main__':
    fire.Fire()