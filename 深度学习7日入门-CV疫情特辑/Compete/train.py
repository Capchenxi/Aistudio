#!/usr/bin/env python
# coding: utf-8

'''
解压数据集
'''
# !unzip -q -o data/data1917/train_new.zip
# !unzip -q -o data/data1917/test_new.zip


# In[2]:


'''
加载相关类库
'''
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
from utile import *
from model import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
查看train.json相关信息，重点关注annotations中的标注信息
'''
f = open('/home/aistudio/data/data1917/train.json', encoding='utf-8')
content = json.load(f)

'''
将上面的到的content中的name中的“stage1/”去掉
'''
for j in range(len(content['annotations'])):
    content['annotations'][j]['name'] = content['annotations'][j]['name'].lstrip('stage1').lstrip('/')

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
    size_x, size_y = img.size
    train_img_size = (640, 480)
    img = img.resize(train_img_size, Image.ANTIALIAS)
    img = np.array(img)
    img = img / 255.0

    gt = []
    for b_l in range(len(ann)):
        # 假设人体是使用方框标注的，通过求均值的方法将框变为点
        if 'w' in ann[b_l].keys():
            x = (ann[b_l]['x'] + (ann[b_l]['x'] + ann[b_l]['w'])) / 2
            y = ann[b_l]['y'] + 20  # ?
            x = (x * 640 / size_x) / 8
            y = (y * 480 / size_y) / 8
            gt.append((x, y))
        else:
            x = ann[b_l]['x']
            y = ann[b_l]['y']
            x = (x * 640 / size_x) / 8  # why divided by 8?
            y = (y * 480 / size_y) / 8
            gt.append((x, y))
            # Return img and points list
    return img, gt


'''
密度图处理
'''


def ground(img, gt):
    imgs = img
    x = imgs.shape[0] / 8
    y = imgs.shape[1] / 8
    k = np.zeros((int(x), int(y)))

    for i in range(0, len(gt)):
        if int(gt[i][1]) < int(x) and int(gt[i][0]) < int(y):
            k[int(gt[i][1]), int(gt[i][0])] = 1

    k = gaussian_filter_density(k)
    return k


'''
定义数据生成器
'''


def train_set():
    def inner():
        for ig_index in range(2000):  # 遍历所有图片
            if len(content['annotations'][ig_index]['annotation']) == 2: continue  # ?
            if len(content['annotations'][ig_index]['annotation']) == 3: continue
            if content['annotations'][ig_index]['ignore_region']:  # 把忽略区域都用像素为0填上
                ig_list = []  # 存放忽略区1的数据
                ig_list1 = []  # 存放忽略区2的数据
                # print(content['annotations'][ig_index]['ignore_region'])
                if len(content['annotations'][ig_index]['ignore_region']) == 1:  # 因为每张图的忽略区域最多2个，这里是为1的情况
                    # print('ig1',ig_index)
                    ign_rge = content['annotations'][ig_index]['ignore_region'][0]  # 取第一个忽略区的数据
                    for ig_len in range(len(ign_rge)):  # 遍历忽略区坐标个数，组成多少变型
                        ig_list.append([ign_rge[ig_len]['x'], ign_rge[ig_len]['y']])  # 取出每个坐标的x,y然后组成一个小列表放到ig_list
                    ig_cv_img = cv2.imread(content['annotations'][ig_index]['name'])  # 用cv2读取一张图片
                    pts = np.array(ig_list, np.int32)  # 把ig_list转成numpy.ndarray数据格式，为了填充需要
                    cv2.fillPoly(ig_cv_img, [pts], (0, 0, 0), cv2.LINE_AA)  # 使用cv2.fillPoly方法对有忽略区的图片用像素为0填充

                    ig_img = Image.fromarray(cv2.cvtColor(ig_cv_img, cv2.COLOR_BGR2RGB))  # cv2转PIL

                    ann = content['annotations'][ig_index]['annotation']  # 把所有标注的信息读取出来

                    ig_im, gt = picture_opt(ig_img, ann)
                    k = ground(ig_im, gt)

                    groundtruth = np.asarray(k)
                    groundtruth = groundtruth.T.astype('float32')
                    ig_im = ig_im.transpose().astype('float32')
                    yield ig_im, groundtruth

                if len(content['annotations'][ig_index]['ignore_region']) == 2:  # 有2个忽略区域
                    # print('ig2',ig_index)
                    ign_rge = content['annotations'][ig_index]['ignore_region'][0]
                    ign_rge1 = content['annotations'][ig_index]['ignore_region'][1]
                    for ig_len in range(len(ign_rge)):
                        ig_list.append([ign_rge[ig_len]['x'], ign_rge[ig_len]['y']])
                    for ig_len1 in range(len(ign_rge1)):
                        ig_list1.append([ign_rge1[ig_len1]['x'], ign_rge1[ig_len1]['y']])
                    ig_cv_img2 = cv2.imread(content['annotations'][ig_index]['name'])
                    pts = np.array(ig_list, np.int32)
                    pts1 = np.array(ig_list1, np.int32)
                    cv2.fillPoly(ig_cv_img2, [pts], (0, 0, 0), cv2.LINE_AA)
                    cv2.fillPoly(ig_cv_img2, [pts1], (0, 0, 0), cv2.LINE_AA)

                    ig_img2 = Image.fromarray(cv2.cvtColor(ig_cv_img2, cv2.COLOR_BGR2RGB))  # cv2转PIL

                    ann = content['annotations'][ig_index]['annotation']  # 把所有标注的信息读取出来

                    ig_im, gt = picture_opt(ig_img2, ann)
                    k = ground(ig_im, gt)
                    k = np.zeros((int(ig_im.shape[0] / 8), int(ig_im.shape[1] / 8)))

                    groundtruth = np.asarray(k)
                    groundtruth = groundtruth.T.astype('float32')
                    ig_im = ig_im.transpose().astype('float32')
                    yield ig_im, groundtruth

            else:
                img = Image.open(content['annotations'][ig_index]['name'])
                ann = content['annotations'][ig_index]['annotation']  # 把所有标注的信息读取出来

                im, gt = picture_opt(img, ann)
                k = ground(im, gt)

                groundtruth = np.asarray(k)
                groundtruth = groundtruth.T.astype('float32')
                im = im.transpose().astype('float32')
                yield im, groundtruth

    return inner


BATCH_SIZE = 10  # 每次取3张
# 设置训练reader
train_reader = paddle.batch(
    paddle.reader.shuffle(
        train_set(), buf_size=512),
    batch_size=BATCH_SIZE)


class ConvPool(fluid.dygraph.Layer):
    '''卷积+池化'''

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 groups,
                 pool_padding=0,
                 pool_type='max',
                 conv_stride=1,
                 conv_padding=0,
                 act=None):
        super(ConvPool, self).__init__()

        self._conv2d_list = []

        for i in range(groups):
            conv2d = self.add_sublayer(  # 返回一个由所有子层组成的列表。
                'bb_%d' % i,
                fluid.dygraph.Conv2D(
                    num_channels=num_channels,  # 通道数
                    num_filters=num_filters,  # 卷积核个数
                    filter_size=filter_size,  # 卷积核大小
                    stride=conv_stride,  # 步长
                    padding=conv_padding,  # padding大小，默认为0
                    act=act)
            )
        self._conv2d_list.append(conv2d)

        self._pool2d = fluid.dygraph.Pool2D(
            pool_size=pool_size,  # 池化核大小
            pool_type=pool_type,  # 池化类型，默认是最大池化
            pool_stride=pool_stride,  # 池化步长
            pool_padding=1  # 填充大小
        )

        self._batchnorm = fluid.dygraph.BatchNorm(num_filters)

    def forward(self, inputs):
        x = inputs
        for conv in self._conv2d_list:
            x = conv(x)
        x = self._pool2d(x)
        x = self._batchnorm(x)

        return x


class DDCB(fluid.dygraph.Layer):
    def __init__(self, input_size):
        super(DDCB, self).__init__()

        self.red1 = fluid.dygraph.Conv2D(num_channels=input_size, num_filters=256, filter_size=1, dilation=1,
                                         act='relu')
        self.pink1 = fluid.dygraph.Conv2D(num_channels=256, num_filters=64, filter_size=3, padding=1, dilation=1,
                                          act='relu')
        self.red2 = fluid.dygraph.Conv2D(num_channels=64 + input_size, num_filters=256, filter_size=1, dilation=1,
                                         act='relu')
        self.brown1 = fluid.dygraph.Conv2D(num_channels=256, num_filters=64, filter_size=3, padding=2, dilation=2,
                                           act='relu')
        self.red3 = fluid.dygraph.Conv2D(num_channels=64 * 2 + input_size, num_filters=256, filter_size=1, dilation=1,
                                         act='relu')
        self.green1 = fluid.dygraph.Conv2D(num_channels=256, num_filters=64, filter_size=3, padding=3, dilation=3,
                                           act='relu')
        self.pink2 = fluid.dygraph.Conv2D(num_channels=64 * 2 + input_size, num_filters=512, filter_size=3, padding=1,
                                          dilation=1, act='relu')

    def forward(self, input):
        # print(input.shape)

        x_red1 = self.red1(input)
        # print(x_red1.shape)
        x_pink1 = self.pink1(x_red1)
        # print(x_pink1.shape)
        x_c1 = fluid.layers.concat([input, x_pink1], axis=1)  ####
        # print(x_c1.shape)

        x_red2 = self.red2(x_c1)
        # print(x_red2.shape)
        x_brown1 = self.brown1(x_red2)
        # print(x_brown1.shape)
        x_c2 = fluid.layers.concat([x_brown1, x_pink1, input], axis=1)  ####
        # print(x_c2.shape)

        x_red3 = self.red3(x_c2)
        x_green1 = self.green1(x_red3)

        x_c3 = fluid.layers.concat([x_green1, x_brown1, input], axis=1)  ####

        y = self.pink2(x_c3)

        return y


class MyNet(fluid.dygraph.Layer):
    '''
    网络
    '''

    def __init__(self):
        super(MyNet, self).__init__()
        # [-1, 3, 640, 480]
        self.conv1 = ConvPool(num_channels=3, num_filters=64, filter_size=3, pool_size=3, conv_padding=1, pool_stride=2,
                              groups=2, act='relu')
        # [-1, 64, 320, 240]
        self.conv2 = ConvPool(num_channels=64, num_filters=128, filter_size=3, pool_size=3, conv_padding=1,
                              pool_stride=2, groups=2, act='relu')
        # [-1, 128, 160, 120]
        self.conv3 = ConvPool(num_channels=128, num_filters=256, filter_size=3, pool_size=3, conv_padding=1,
                              pool_stride=2, groups=3, act='relu')
        # [-1, 256, 80, 60]
        self.conv4 = ConvPool(num_channels=256, num_filters=512, filter_size=3, pool_size=3, conv_padding=1,
                              pool_stride=1, groups=3, act='relu')
        # [-1, 512, 40, 30]
        self.DDCB1 = DDCB(input_size=512)
        self.DDCB2 = DDCB(input_size=1024)
        self.DDCB3 = DDCB(input_size=512 * 3)

        self.pink1 = fluid.dygraph.Conv2D(num_channels=512 * 4, num_filters=128, filter_size=3, padding=1, dilation=1,
                                          act='relu')
        self.pink2 = fluid.dygraph.Conv2D(num_channels=128, num_filters=64, filter_size=3, padding=1, dilation=1,
                                          act='relu')

        self.dw = fluid.dygraph.Conv2D(num_channels=64, num_filters=1, filter_size=1, dilation=1)

    def forward(self, inputs, label=None):
        """前向计算"""
        # print(inputs.shape)
        x1 = self.conv1(inputs)
        # print(x1.shape)
        x2 = self.conv2(x1)
        # print(x2.shape)
        x3 = self.conv3(x2)
        # print(x3.shape)
        x4 = self.conv4(x3)
        # print(x4.shape)
        # print('start')

        x5 = self.DDCB1(x4)
        # print(x5.shape)
        x6 = self.DDCB2(fluid.layers.concat([x4, x5], axis=1))
        # print(x6.shape)
        x7 = self.DDCB3(fluid.layers.concat([x4, x5, x6], axis=1))
        # print(x7.shape)

        x8 = self.pink1(fluid.layers.concat([x4, x5, x6, x7], axis=1))
        # print(x8.shape)
        x9 = self.pink2(x8)
        # print(x9.shape)

        y = self.dw(x9)

        return y


'''
模型训练
'''
with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
    model, _ = fluid.dygraph.load_dygraph("MyNet_DDCB")
    net = MyNet()
    net.load_dict(model)
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.00001, parameter_list=net.parameters())
    for epoch_num in range(1):
        for batch_id, data in enumerate(train_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([x[1] for x in data]).astype('float32')
            y_data = y_data[:, np.newaxis]

            # 将Numpy转换为DyGraph接收的输入
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            label.stop_gradient = True
            # print(img.shape, label.shape)

            out = net(img, label)

            # print(out.shape) #[-1, 1, 80, 60]
            loss = fluid.layers.square_error_cost(out, label)
            avg_loss = fluid.layers.mean(loss)

            # 使用backward()方法可以执行反向网络
            avg_loss.backward()
            optimizer.minimize(avg_loss)

            # 将参数梯度清零以保证下一轮训练的正确性
            net.clear_gradients()

            dy_param_value = {}
            for param in net.parameters():
                dy_param_value[param.name] = param.numpy

            if batch_id % 10 == 0:
                print("Loss at epoch {} step {}: {}".format(epoch_num, batch_id, avg_loss.numpy()))
    # 保存模型参数
    fluid.save_dygraph(net.state_dict(), "MyNet_DDCB")
    print("Final loss: {}".format(avg_loss.numpy()))

# In[ ]:


fluid.save_dygraph(net.state_dict(), "MyNet_DDCB")
print("Final loss: {}".format(avg_loss.numpy()))

# # 五、模型预测

# In[14]:


data_dict = {}

'''
模型预测
'''
with fluid.dygraph.guard():
    model, _ = fluid.dygraph.load_dygraph("MyNet_DDCB")
    net = MyNet()
    net.load_dict(model)
    net.eval()

    # 获取预测图片列表
    test_zfile = zipfile.ZipFile("/home/aistudio/data/data1917/test_new.zip")
    l_test = []
    for test_fname in test_zfile.namelist()[1:]:
        l_test.append(test_fname)

    for index in range(len(l_test)):
        test_img = Image.open(l_test[index])
        test_img = test_img.resize((640, 480))
        test_im = np.array(test_img)
        test_im = test_im / 255.0
        test_im = test_im.transpose().reshape(3, 640, 480).astype('float32')
        l_test[index] = l_test[index].lstrip('test').lstrip('/')

        dy_x_data = np.array(test_im).astype('float32')
        dy_x_data = dy_x_data[np.newaxis, :, :, :]
        img = fluid.dygraph.to_variable(dy_x_data)
        out = net(img)
        temp = out[0][0]
        temp = temp.numpy()
        people = np.sum(temp)
        data_dict[l_test[index]] = int(people)

import csv

with open('results.csv', 'w') as csvfile:
    fieldnames = ['id', 'predicted']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for k, v in data_dict.items():
        writer.writerow({'id': k, 'predicted': v})
print("结束")