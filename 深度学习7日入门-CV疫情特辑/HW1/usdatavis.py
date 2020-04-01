import json
import re
import requests
import datetime

from pyecharts.charts import Map
from pyecharts.faker import Faker
from pyecharts.datasets import register_url
from pyecharts import options as opts

#Get data using api
url = "https://covid19-data.p.rapidapi.com/us"

headers = {
    'x-rapidapi-host': "covid19-data.p.rapidapi.com",
    'x-rapidapi-key': "38fbbbe653msh90ba1b3a0a51a96p1d0eb5jsn08a2f37b8277"
    }

# Data organization
response = requests.request("GET", url, headers=headers)
text = response.content.decode()
names = re.findall(r'"state":"(.*?)",', text)
cases = re.findall(r'"confirmed":(.*?),', text)
latest = {}

for n in names:
    if n not in latest:
        latest[n] = cases[names.index(n)]
    else:
        continue
data_states = []
data_cases = []

for k, v in latest.items():
    data_states.append(k)
    data_cases.append(int(v))

pieces = [
    {'min': 10000, 'color': '#540d0d'},
    {'max': 9999, 'min': 1000, 'color': '#9c1414'},
    {'max': 999, 'min': 500, 'color': '#d92727'},
    {'max': 499, 'min': 100, 'color': '#ed3232'},
    {'max': 99, 'min': 10, 'color': '#f27777'},
    {'max': 9, 'min': 1, 'color': '#f7adad'},
    {'max': 0, 'color': '#f7e4e4'},
]

register_url("https://echarts-maps.github.io/echarts-countries-js/")
m = Map()
m.add("累计确诊",  [list(z) for z in zip(data_states, data_cases)], "美国")
m.set_series_opts(label_opts=opts.LabelOpts(font_size=12),
                  is_show=False)
m.set_global_opts(
                title_opts=opts.TitleOpts(title="美国疫情地图", subtitle="数据有些奇怪"),
                legend_opts=opts.LegendOpts(is_show=False),
                 visualmap_opts=opts.VisualMapOpts(pieces=pieces,
                                                    is_piecewise=True,   #是否为分段型
                                                    is_show=True))
m.render('./data/美国疫情地图.html')