#coding:utf-8
import os
import random
import sys
import time
import h5py
import cv2
import numpy as np
from multiprocessing import Pool
import linecache


# 影像文件夹所在目录
img_root = 'E:/work/data/landmark/samples/98/'
# 训练数据txt路径
train_path = 'E:/work/data/landmark/samples/98/train_98_list.txt'
# 输出路径
train_out = './hdf5_ldmark_train.h5'

# 将txt中的数据存入
with open(train_path) as f:
    lines = f.readlines()

max_img_len = min(500, len(lines))
file_list = []  # 存入影像路径
# 建立标签和数据数组
# 若要生成hdf5数据，必须先把影像和标签变为数组
# 本文标签数目为2，影像数据：channel = 3,width = 256,height = 256故生成如下形式数据
labels = np.zeros((int(max_img_len), 196)).astype(np.float32)
datas = np.zeros((int(max_img_len), 3, 112, 112)).astype(np.float32)
# 读取数据
count = 0
for idx in range(max_img_len):
    a = lines[idx].split()
    data = map(float, a[1:197])
    file_list.append(a[0])
    for idx_pt in range(196):
        labels[count][idx_pt] = data[idx_pt]
    count += 1
f.close()

# caffe利用hfd5数据时，在输入层没有transform_param 参数，所以需要先对影像数据进行预处理
for i, file in enumerate(file_list):
    path = os.path.join(img_root, file)
    image = cv2.imread(path)  # 获取影像
    image = cv2.resize(image, (112, 112))  # 重采样为256*256大小的图像
    img = np.array(image)
    img = img.transpose(2, 0, 1)  # 讲图像从宽 高 通道 形式转化为通道 宽 高  caffe读取图像形式
    datas[i, :, :, :] = img.astype(np.float32)  # hdf5要求数据为float或double形式

# 保存hdf5文件
with h5py.File(train_out, 'w') as fout:
    # 'data'必须和train_val.prototxt文件里数据层中top：后边的名称一致，在修改prototxt文件时会进一步说明
    fout.create_dataset('data', data=datas)
    fout.create_dataset('label', data=labels)
fout.close()