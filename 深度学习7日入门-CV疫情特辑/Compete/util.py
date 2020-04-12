import zipfile
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import matplotlib.image as mping

import json
import numpy as np
import cv2
import sys
import time
import h5py
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
from matplotlib import cm as CM
from paddle.utils.plot import Ploter
from PIL import Image
from PIL import ImageFile

'''
使用高斯滤波变换生成密度图
'''


def gaussian_filter_density(gt):
    # 初始化密度图
    density = np.zeros(gt.shape, dtype=np.float32)

    # 获取gt中不为0的元素的个数
    gt_count = np.count_nonzero(gt)

    # 如果gt全为0，就返回全0的密度图
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            # sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            sigma = 25
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density

'''
图片操作：对图片进行resize、归一化，将方框标注变为点标注
返回：resize后的图片 和 gt
'''
def picture_opt(img, ann):
    size_x,size_y = img.size
    train_img_size = (640,480)
    img = img.resize(train_img_size,Image.ANTIALIAS)
    img = np.array(img)
    img = img / 255.0

    gt = []
    for b_l in range(len(ann)):
        # 假设人体是使用方框标注的，通过求均值的方法将框变为点
        if 'w' in ann[b_l].keys():
            x = (ann[b_l]['x']+(ann[b_l]['x']+ann[b_l]['w']))/2
            y = ann[b_l]['y']+20 # ?
            x = (x*640/size_x)/8
            y = (y*480/size_y)/8
            gt.append((x,y))
        else:
            x = ann[b_l]['x']
            y = ann[b_l]['y']
            x = (x*640/size_x)/8 # why divided by 8?
            y = (y*480/size_y)/8
            gt.append((x,y))
    # Return img and points list
    return img,gt

'''
密度图处理
'''
def ground(img,gt):
    imgs = img
    x = imgs.shape[0]/8
    y = imgs.shape[1]/8
    k = np.zeros((int(x),int(y)))

    for i in range(0,len(gt)):
        if int(gt[i][1]) < int(x) and int(gt[i][0]) < int(y):
            k[int(gt[i][1]),int(gt[i][0])]=1

    k = gaussian_filter_density(k)
    return k
