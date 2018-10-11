## Hand Pose Estimation 手部姿态估计

手部姿态估计，Tensorflow代码实现，主要实现手部21关键点2D坐标回归



### 环境配置

- Tensorflow = 1.4.0  (cuda8, cudnn6)

- Python >= 3.5

- 其他：opencv 3.1.0

- 辅助工具包： fire, ipdb, jupyterlab, zsh, tmux等

  以上环境均封装在docker中：

  `dl-registry.service.163.org:5000/handpose/tensorflow:1.4-cuda8-cudnn6`

### 运行说明

#### Step1 拉取docker镜像
```
docker pull dl-registry.service.163.org:5000/handpose/tensorflow:1.4-cuda8-cudnn6 
```

#### Step2 启动docker
- `-p`参数为container预留端口映射，方便在host对应端口查看tensorboard/jupyterlab等
- `-v`参数挂载host文件夹到container中，代码区workspace及数据data分开挂载
- 容器使用及后台程序呢运行，参照[tips](https://g.hz.netease.com/Gesture/hand_pose_tensorflow/wikis/home#%E4%B8%89tips)

```
for example:
docker run --runtime=nvidia --name=container_name -p 10000:10000 \
	-p 10001:10001 -v /home/username/workspace:/root/workspace \
	-v /srv/data0/gesture/handpose/data:/root/workspace/data \
	-it dl-registry.service.163.org:5000/handpose/tensorflow:1.4-cuda8-cudnn6
```

#### Step3 运行（训练、验证、测试）
- 均通过运行主函数handpose_tf.py来运行，使用[fire](https://blog.csdn.net/u014102846/article/details/77946592)模块调用不同函数。
- 建议建立参考文件目录结构，[tips](https://g.hz.netease.com/Gesture/hand_pose_tensorflow/wikis/home#3-%E6%96%87%E4%BB%B6%E6%A0%91%E7%8A%B6%E7%BB%93%E6%9E%84)
- 默认参数配置项位于`config.py`，不同过程建议的参数项不同，[tips](https://g.hz.netease.com/Gesture/hand_pose_tensorflow/wikis/home#1-config%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)

```
Usage: python handpose_tf.py train/evaluation/test \
			--use_gpu=True \
			--gpu_id=0(单卡) \
			--[other_option=...] (详细参见config.py中配置)
			
for example:
train:
python handpose_tf.py train --train_marker=1 --dataset_type=RHD \
		--max_iter=50000 --with_dnst=True --augument=False
		
evaluation:
python handpose_tf.py evaluation --dataset_type=RHD --with_dnst=True \
		--load_path=0810_n2_RHD_c1.2_b8_aFalse_vTrue_dTrue_i3.0w
		
test:
python handpose_tf.py test --test_data_path='hand_pose_test' --plot_eval=True \
		--load_path=0810_n2_RHD_c1.2_b8_aFalse_vTrue_dTrue_i3.0w
```

#### Step 数据转储成TFRecords（可选）
- 主要使用数据集：`RHD` 、`STB_320`(STB数据裁剪到320)、`RS_320`(前两者混合)
- 主要使用脚本为`convert_to_tfrecords.py`，数据树状结构参考[tips](https://g.hz.netease.com/Gesture/hand_pose_tensorflow/wikis/home#3-%E6%96%87%E4%BB%B6%E6%A0%91%E7%8A%B6%E7%BB%93%E6%9E%84)

```
Usage: python convert_to_tfrecords.py convert_to_TFRecords \
		--dataset_list=RHD[or STB_320 or RHD/STB_320] --state=train[or eval]
```