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
            pool_padding=pool_padding  # 填充大小
        )

        self._batchnorm = fluid.dygraph.BatchNorm(num_filters)

    def forward(self, inputs):
        x = inputs
        for conv in self._conv2d_list:
            x = conv(x)
        x = self._pool2d(x)
        x = self._batchnorm(x)

        return x


class MyVGG(fluid.dygraph.Layer):
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
                              pool_stride=2, groups=3, act='relu')
        # [-1, 512, 40, 30]

    def forward(self, inputs, label=None):
        """前向计算"""
        x1 = self.conv1(inputs)
        # print(x1.shape)
        x2 = self.conv2(x1)
        # print(x2.shape)
        x3 = self.conv3(x2)
        # print(x3.shape)
        x4 = self.conv4(x3)

        return x4


class MyDSnet(fluid.dygraph.Layer):
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
