#!/usr/bin/env python
# coding: utf-8

# # 作业说明：
# 
# **作业1:飞桨本地安装**
# 
# **提交飞桨本地安装成功的截图**，如下图所示，有安装问题可以随时在群里提问，可参考飞桨官网：https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/c93925b52b2f4f9dbeb62569122924f9ee3b6c69c3b14219a74425d8360d6639" height="200" width="400" />
# 
# 
# <br/>
# <br/>
# 
# **作业2：新冠疫情可视化**
# 
# 请根据课上所学内容，爬取3月31日当天丁香园公开的统计数据，根据累计确诊数，使用pyecharts绘制疫情分布图，如下图所示，**提交截图。**
# 
# Pycharts api可参考：https://pyecharts.org/#/zh-cn/
# 
# 
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/24a15aae792b49ecb4b11aa30530e97691a092ef2af94f9f97751826096b4cc8" height="200" width="400" />
# 
# 
# **重要：**
# 
# **一定要用正确的姿势在Notebook上传图片+生成版本作业提交示例：** https://aistudio.baidu.com/aistudio/projectDetail/296022
# 
# 
# 
# 
# 
# 

# 可视化，是一种利用计算机图形学和图像处理技术，将数据转换成图像在屏幕上显示出来，再进行交互处理的理论、方法和技术。
# 
# **本次实践基于丁香园公开的统计数据，实现新冠疫情可视化，包括疫情地图、疫情增长趋势图、疫情分布图等。**
# <br />
# <br />
# 
# **全国疫情地图如下：**
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/3ca5fa8e7019498ab7217aea7a552f0394ff092045174b9284920dbee57b0c1a" height="500" width="500" />
# 
# <br />
# <br />
# <br />
# <br />
# 
# **疫情增长趋势图如下：**
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/28000edb58d34e2e991b6dc2739007ae837b9b031d8842528c9d9506941fbd17" height="500" width="600" />
# 
# 
# 
# 
# 
# 

# # 一、数据准备
# 
# 上网的全过程:
# 
# - 普通用户
# 
#  打开浏览器 --> 往目标站点发送请求 --> 接收响应数据 --> 渲染到页面上。
#                  
# - 爬虫程序
# 
# 模拟浏览器 --> 往目标站点发送请求 --> 接收响应数据 --> 提取有用的数据 --> 保存到本地/数据库。
# 
# <br />
# 
# 爬虫的过程：
# 
# 1.发送请求（requests模块）
# 
# 2.获取响应数据（服务器返回）
# 
# 3.解析并提取数据（re正则）
# 
# 4.保存数据
# 

# **request模块：**
# 
# requests是python实现的简单易用的HTTP库，官网地址：http://cn.python-requests.org/zh_CN/latest/
# 
# **re模块：**
# 
# re模块是python用于匹配字符串的模块，该模块中提供的很多功能是基于正则表达式实现的，

# In[1]:


import json
import re
import requests
import datetime

today = datetime.date.today().strftime('%Y%m%d')   #20200331

def crawl_dxy_data():
    """
    爬取丁香园实时统计数据，保存到data目录下，以当前日期作为文件名，存JSON文件
    """
    response = requests.get('https://ncov.dxy.cn/ncovh5/view/pneumonia') #request.get()用于请求目标网站
    print(response.status_code)                                          # 打印状态码


    try:
        url_text = response.content.decode()                             #更推荐使用response.content.deocde()的方式获取响应的html页面
        print(url_text)
        url_content = re.search(r'window.getAreaStat = (.*?)}]}catch',   #re.search():扫描字符串以查找正则表达式模式产生匹配项的第一个位置 ，然后返回相应的match对象。
                                url_text, re.S)                          #在字符串a中，包含换行符\n，在这种情况下：如果不使用re.S参数，则只在每一行内进行匹配，如果一行没有，就换下一行重新开始;
                                                                         #而使用re.S参数以后，正则表达式会将这个字符串作为一个整体，在整体中进行匹配。
        texts = url_content.group()                                      #获取匹配正则表达式的整体结果
        content = texts.replace('window.getAreaStat = ', '').replace('}catch', '') #去除多余的字符
        json_data = json.loads(content)                                         
        with open('data/' + today + '.json', 'w', encoding='UTF-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
    except:
        print('<Response [%s]>' % response.status_code)


def crawl_statistics_data():
    """
    获取各个省份历史统计数据，保存到data目录下，存JSON文件
    """
    with open('data/'+ today + '.json', 'r', encoding='UTF-8') as file:
        json_array = json.loads(file.read())

    statistics_data = {}
    for province in json_array:
        response = requests.get(province['statisticsData'])
        try:
            statistics_data[province['provinceShortName']] = json.loads(response.content.decode())['data']
        except:
            print('<Response [%s]> for url: [%s]' % (response.status_code, province['statisticsData']))

    with open("data/statistics_data.json", "w", encoding='UTF-8') as f:
        json.dump(statistics_data, f, ensure_ascii=False)


if __name__ == '__main__':
    crawl_dxy_data()
    crawl_statistics_data()


# In[2]:


'''
安装第三方库pyecharts ，如果下载时出现断线和速度过慢的问题导致下载失败，可以尝试使用清华镜像
'''
#!pip install pyecharts
get_ipython().system('pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyecharts')


# # 二、疫情地图
# 
# Echarts 是一个由百度开源的数据可视化工具，凭借着良好的交互性，精巧的图表设计，得到了众多开发者的认可。而 Python 是一门富有表达力的语言，很适合用于数据处理。当数据分析遇上数据可视化时，**pyecharts** 诞生了。pyecharts api可以参考：https://pyecharts.org/#/zh-cn/chart_api
# 
# <br />
# <br/>
# 
# **使用 options 配置项，在 pyecharts 中，一切皆 Options。**
# 
# 主要分为全局配置组件和系列配置组件。
# 
# （1）系列配置项 set_series_opts(),可配置图元样式、文字样式、标签样式、点线样式等；   
# 
# （2）全局配置项 set_global_opts()，可配置标题、动画、坐标轴、图例等;                       
# 
# 先来认识下全局配置组件吧
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/dc6f8d74e83a49ccb2ce5d0c2d8518fc7e731d1d203a4ec4ace84d6aacf73910" height="600" width="700" />
# 
# 
# 

# # 2.1全国疫情地图

# In[3]:


import json
import datetime
from pyecharts.charts import Map
from pyecharts import options as opts

# 读原始数据文件
today = datetime.date.today().strftime('%Y%m%d')   #20200331
datafile = 'data/'+ today + '.json'
with open(datafile, 'r', encoding='UTF-8') as file:
    json_array = json.loads(file.read())

# 分析全国实时确诊数据：'confirmedCount'字段
china_data = []
for province in json_array:
    china_data.append((province['provinceShortName'], province['confirmedCount']))
china_data = sorted(china_data, key=lambda x: x[1], reverse=True)                 #reverse=True,表示降序，反之升序

print(china_data)
# 全国疫情地图
# 自定义的每一段的范围，以及每一段的特别的样式。
pieces = [
    {'min': 10000, 'color': '#540d0d'},
    {'max': 9999, 'min': 1000, 'color': '#9c1414'},
    {'max': 999, 'min': 500, 'color': '#d92727'},
    {'max': 499, 'min': 100, 'color': '#ed3232'},
    {'max': 99, 'min': 10, 'color': '#f27777'},
    {'max': 9, 'min': 1, 'color': '#f7adad'},
    {'max': 0, 'color': '#f7e4e4'},
]
labels = [data[0] for data in china_data]
counts = [data[1] for data in china_data]

m = Map()
m.add("累计确诊", [list(z) for z in zip(labels, counts)], 'china')

#系列配置项,可配置图元样式、文字样式、标签样式、点线样式等
m.set_series_opts(label_opts=opts.LabelOpts(font_size=12),
                  is_show=False)
#全局配置项,可配置标题、动画、坐标轴、图例等
m.set_global_opts(title_opts=opts.TitleOpts(title='全国实时确诊数据',
                                            subtitle='数据来源：丁香园'),
                  legend_opts=opts.LegendOpts(is_show=False),
                  visualmap_opts=opts.VisualMapOpts(pieces=pieces,
                                                    is_piecewise=True,   #是否为分段型
                                                    is_show=True))       #是否显示视觉映射配置
#render（）会生成本地 HTML 文件，默认会在当前目录生成 render.html 文件，也可以传入路径参数，如 m.render("mycharts.html")
m.render(path='/home/aistudio/data/全国实时确诊数据.html')


# # 2.2湖北省疫情地图

# In[4]:


import json
import datetime
from pyecharts.charts import Map
from pyecharts import options as opts

# 读原始数据文件
today = datetime.date.today().strftime('%Y%m%d')   #20200315
datafile = 'data/'+ today + '.json'
with open(datafile, 'r', encoding='UTF-8') as file:
    json_array = json.loads(file.read())

# 分析湖北省实时确诊数据
# 读入规范化的城市名称，用于规范化丁香园数据中的城市简称
with open('/home/aistudio/data/data24815/pycharts_city.txt', 'r', encoding='UTF-8') as f:
    defined_cities = [line.strip() for line in f.readlines()]


def format_city_name(name, defined_cities):
    for defined_city in defined_cities:
        if len((set(defined_city) & set(name))) == len(name):
            name = defined_city
            if name.endswith('市') or name.endswith('区') or name.endswith('县') or name.endswith('自治州'):
                return name
            return name + '市'
    return None


province_name = '湖北'
for province in json_array:
    if province['provinceName'] == province_name or province['provinceShortName'] == province_name:
        json_array_province = province['cities']
        hubei_data = [(format_city_name(city['cityName'], defined_cities), city['confirmedCount']) for city in
                      json_array_province]
        hubei_data = sorted(hubei_data, key=lambda x: x[1], reverse=True)

        print(hubei_data)

labels = [data[0] for data in hubei_data]
counts = [data[1] for data in hubei_data]
pieces = [
    {'min': 10000, 'color': '#540d0d'},
    {'max': 9999, 'min': 1000, 'color': '#9c1414'},
    {'max': 999, 'min': 500, 'color': '#d92727'},
    {'max': 499, 'min': 100, 'color': '#ed3232'},
    {'max': 99, 'min': 10, 'color': '#f27777'},
    {'max': 9, 'min': 1, 'color': '#f7adad'},
    {'max': 0, 'color': '#f7e4e4'},
]

m = Map()
m.add("累计确诊", [list(z) for z in zip(labels, counts)], '湖北')
m.set_series_opts(label_opts=opts.LabelOpts(font_size=12),
                  is_show=False)
m.set_global_opts(title_opts=opts.TitleOpts(title='湖北省实时确诊数据',
                                            subtitle='数据来源：丁香园'),
                  legend_opts=opts.LegendOpts(is_show=False),
                  visualmap_opts=opts.VisualMapOpts(pieces=pieces,
                                                    is_piecewise=True,
                                                    is_show=True))
m.render(path='/home/aistudio/data/湖北省实时确诊数据.html')


# # 三、疫情增长趋势图

# In[5]:


import numpy as np
import json
from pyecharts.charts import Line
from pyecharts import options as opts

# 读原始数据文件
datafile = 'data/statistics_data.json'
with open(datafile, 'r', encoding='UTF-8') as file:
    json_dict = json.loads(file.read())

# 分析各省份2月1日至今的新增确诊数据：'confirmedIncr'
statistics__data = {}
for province in json_dict:
    statistics__data[province] = []
    for da in json_dict[province]:
        if da['dateId'] >= 20200201:
            statistics__data[province].append(da['confirmedIncr'])

# 获取日期列表
dateId = [str(da['dateId'])[4:6] + '-' + str(da['dateId'])[6:8] for da in json_dict['湖北'] if
          da['dateId'] >= 20200201]

# 全国新增趋势
all_statis = np.array([0] * len(dateId))
for province in statistics__data:
    all_statis = all_statis + np.array(statistics__data[province])

all_statis = all_statis.tolist()
# 湖北新增趋势
hubei_statis = statistics__data['湖北']
# 湖北以外的新增趋势
other_statis = [all_statis[i] - hubei_statis[i] for i in range(len(dateId))]

line = Line()
line.add_xaxis(dateId)
line.add_yaxis("全国新增确诊病例",   #图例
                all_statis,       #数据
                is_smooth=True,   #是否平滑曲线
               linestyle_opts=opts.LineStyleOpts(width=4, color='#B44038'),#线样式配置项
               itemstyle_opts=opts.ItemStyleOpts(color='#B44038',          #图元样式配置项
                                                 border_color="#B44038",   #颜色
                                                 border_width=10))         #图元的大小
line.add_yaxis("湖北新增确诊病例", hubei_statis, is_smooth=True,
               linestyle_opts=opts.LineStyleOpts(width=2, color='#4E87ED'),
               label_opts=opts.LabelOpts(position='bottom'),              #标签在折线的底部
               itemstyle_opts=opts.ItemStyleOpts(color='#4E87ED',
                                                 border_color="#4E87ED",
                                                 border_width=3))
line.add_yaxis("其他省份新增病例", other_statis, is_smooth=True,
               linestyle_opts=opts.LineStyleOpts(width=2, color='#F1A846'),
               label_opts=opts.LabelOpts(position='bottom'),              #标签在折线的底部
               itemstyle_opts=opts.ItemStyleOpts(color='#F1A846',
                                                 border_color="#F1A846",
                                                 border_width=3))
line.set_global_opts(title_opts=opts.TitleOpts(title="新增确诊病例", subtitle='数据来源：丁香园'),
                     yaxis_opts=opts.AxisOpts(max_=16000, min_=1, type_="log",    #坐标轴配置项
                                              splitline_opts=opts.SplitLineOpts(is_show=True),#分割线配置项
                                              axisline_opts=opts.AxisLineOpts(is_show=True)))#坐标轴刻度线配置项
line.render(path='/home/aistudio/data/新增确诊趋势图.html')


# # 请在以下cell中完成作业

# # 作业：疫情分布饼状图

# In[35]:


# 引用之前读取的全国疫情数据
print(china_data)
province = [data[0] for data in china_data]
counts = [data[1] for data in china_data]

from pyecharts.charts import Pie
p = Pie(init_opts=opts.InitOpts(width="680px", height="1200px"))
p.add("", [list(z) for z in zip(province, counts)], center=["50%", "35%"])
p.set_global_opts(title_opts=opts.TitleOpts(title="全国实时疫情分布", subtitle="数据来源：丁香园"),
                    legend_opts=opts.LegendOpts(is_show=False))
p.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
p.render(path="/home/aistudio/data/全国实时疫情分布.html")


# # 请将作业截图进行上传

# ![](https://ai-studio-static-online.cdn.bcebos.com/2b0176ee76114d1395bb34183134aa25f7687c168bd2442580d3f5e53d009b4d)
# ![](https://ai-studio-static-online.cdn.bcebos.com/93dc9969e3544d10b764f5a0668e6143c46ac9c6216d4db7983732710a6f9fe7)
# 

# In[ ]:




