# HW1 疫情数据可视化

> 看见这个地图又要哭了，前几天刚过了清明节。
>
> “有的人永远留在了寒冬，为了更多的人可以迎来春天”
>
> 而今天，武汉解禁了。

## 作业任务

这部分的任务要求其实很简单，因为是第一个作业，所以其实主要就是安装paddle paddle，因为我直接在自己的电脑上安装的（是cpu）所以并没有遇到什么GPU的问题。

第二个任务就是把丁香医生的数据爬下来然后可视化在网页中。

重要的就是：

1. 数据获取

   数据获取作业给的教师版本已经给好了，是通过 [丁香园](https://ncov.dxy.cn/ncovh5/view/pneumonia) 网页得到的。主要生成的数据就是全国截至目前各个省的确诊病例，以及湖北省各个市截止目前的确诊病例，以及从每天病例的统计。

2. 利用数据画图

   使用的是百度的 [pyecharts](https://pyecharts.org/#/zh-cn/) 包, 也还挺好用的，原来没有在python上做过地图这样的可视化，不过这个pyecharts的包里面有很多各种各样的图的表示方式。每个函数的api也很清楚，每个都有对应的应用demo

   拿Map举例，主要的调用方式是：

   ```python
   from pyecharts import options as opts
   from pyecharts.charts import Map
   
   m = (
       Map()
       .add("标签", [(x, y)], "china")
       .set_global_opts(title_opts=opts.TitleOpts(title="Map-基本示例"))
       .render("map_base.html")
   )
   
   # 另一种表示方式
   
   m = Map()
   m.add("标签", [(x, y)], "china")
   m.set_global_opts(title_opts=opts.TitleOpts(title="Map-基本示例"))
   m.set_series_opts(title_opts=opts.TitleOpts(title='图标题')
   m.render("保存地址")
   ```

## 一点感想

- 数据获取方面没有太多感想，就是想这次丁香园的数据可以通过截取source里面的关键字来获得，但是有的网页的source中并不会显示过多的信息，可以通过查找你想要的数据有没有开放的api可以供你直接调用的。

- pyechart中有一个不太方便的点是文档里面并没有说 ``` .set_global_opts()``` 输入的值有哪些, 这里我们可以看到的是 ``` title_opts=``` 但是如果需要设置别的全局变量，需要从 [全局变量配置](https://pyecharts.org/#/zh-cn/global_options) 去找需要的关键字。比如设置画布大小，需要在实例化的时候设置。

- 如果想导入世界别的国家的地图，需要下载相关的包。引用的关键字直接用汉语就行:

  ```python
  from pyecharts.datasets import register_url
  
  register_url("https://echarts-maps.github.io/echarts-countries-js/")
  m = Map()
  m.add("累计确诊",  [list(z) for z in zip(data_states, data_cases)], "美国")
  ```

  这样就可以导入美国的地图了。注意数据```data_states```必须是应为的每个州的名称，如果错误的话无法在地图上找到对应的州的位置。

  但如果是国内的省市话，直接 ```m.add("标签", [(x, y)], "郑州")```  这样就可以找到对应省市的地图了。

   