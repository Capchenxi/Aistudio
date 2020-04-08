# 【百度PaddlePaddle】深度学习7日入门-CV疫情
本文件夹为百度Aistudio课程《深度学习7日入门-CV疫情》

- 数据： 从百度Aistudio数据集下载
- 框架： PaddlePaddle
- 语言： Python

作业调试详解及心得体会请参照各作业文件夹。

## 1. 数据可视化[[已完成]]([https://github.com/Capchenxi/Aistudio/tree/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A07%E6%97%A5%E5%85%A5%E9%97%A8-CV%E7%96%AB%E6%83%85%E7%89%B9%E8%BE%91/HW1](https://github.com/Capchenxi/Aistudio/tree/master/深度学习7日入门-CV疫情特辑/HW1))
基于 [丁香园](https://ncov.dxy.cn/ncovh5/view/pneumonia) 整合全国疫情数据更新制作可视化数据图。

作业任务：
	1. 下载安装 PaddlePaddle，课程中有详细介绍
	2. 应用学习pyecharts库学习应用，将全国疫情数据可视化在饼状图中

作业举例：
	1. 已提供爬虫script，将丁香园中的数据下载整合到本地
	2. 已提供pyecharts部分地图和趋势图的实现代码
	
## 2. 构建DNN神经网络手势识别[[已完成]([https://github.com/Capchenxi/Aistudio/tree/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A07%E6%97%A5%E5%85%A5%E9%97%A8-CV%E7%96%AB%E6%83%85%E7%89%B9%E8%BE%91/HW1](https://github.com/Capchenxi/Aistudio/tree/master/深度学习7日入门-CV疫情特辑/HW2))]
基于DNN全连接神经网络对手势和对应数字进行识别。

作业任务：
	1. 完成DNN 神经网络
	2. 优化自己的DNN神经网络(test_acc:0.83, SGD, lr = 0.001, epoch = 200)

## 3. 构建CNN神经网络车牌识别[[已完成]([https://github.com/Capchenxi/Aistudio/tree/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A07%E6%97%A5%E5%85%A5%E9%97%A8-CV%E7%96%AB%E6%83%85%E7%89%B9%E8%BE%91/HW1](https://github.com/Capchenxi/Aistudio/tree/master/深度学习7日入门-CV疫情特辑/HW3))]
基于CNN卷积神经网络对车牌的(数字/ 字母/ 省简称)进行识别

作业任务：
	1. 完成CNN神经网络
	2. 优化自己的CNN神经网络(test_acc:0.93, SGD, lr 0.01, epoch = 200)

## 4. 模块化构建VGG神经网络口罩识别[[已完成]([https://github.com/Capchenxi/Aistudio/tree/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A07%E6%97%A5%E5%85%A5%E9%97%A8-CV%E7%96%AB%E6%83%85%E7%89%B9%E8%BE%91/HW1](https://github.com/Capchenxi/Aistudio/tree/master/深度学习7日入门-CV疫情特辑/HW4))]
定义重复出现的CNN网络结构使其模块化，二分类识别图片中的人是否戴口罩。（更完整的完成任务还需要识别图片中人脸的位置）

作业任务：
	1. 完成CNN神经网络
	2. 优化自己的CNN神经网络（+BatchNorm test_acc=1.0, Adam, lr = 0.0001, epoch=50）
## 5. Paddle Slim 对模型进行精简[已完成]
学习实践Paddle Slim对模型精简模块，可以有效提高模型速度，减小模型所占内存。

作业任务：
	1. 概念选择题
	2. 完成定点量化精简

## 比赛：人流密度检测[得分：0.49469分 排名：51/368]

比赛任务:检测人流密度，得分应该是错误率之类的，越低越好。
训练数据：常规赛-人流密度检测
试验方法：
- Baseline: 老师给出的baseline就已经了不起了，跑通了大概能排到200+左右目前，单纯的cnn放在一起提取的人流密度热力图。的粉黛该是0.68左右
- 我的方法：参考了 [Dense Scale Network for Crowd Counting](https://arxiv.org/pdf/1906.09707.pdf) 这篇文章中密集的残差连接结构，除了MSE作为loss之外还加入了不同尺度下的MAE（mean absolute error）作为loss。
