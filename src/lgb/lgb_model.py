import time
import math
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

'*******************************************读取数据***************************************************'
sales_data = pd.read_csv(r'../../data/car_sales/train_sales_data.csv')
search_data = pd.read_csv(r'../../data/car_sales/train_search_data.csv')

def prepare(data):
    # 对数据进行预处理，将各个属性转为数值特征
    data['date'] = list(map(lambda x, y: str(x)+"."+str(y), data['regYear'], data['regMonth']))
    data['date'] = pd.to_datetime(data['date'])
    if 'forecastVolum' in list(data.columns):
        data = data.drop(['forecastVolum'], axis=1)

    if 'province' in list(data.columns):
        pro_label = dict(zip(sorted(list(set(data['province']))), range(0, len(set(data['province'])))))
        data['pro_id'] = data['province'].map(pro_label)
        data = data.drop(['adcode', 'province'], axis=1)

    if 'bodyType' in list(data.columns):
       body_label = dict(zip(sorted(list(set(data['bodyType']))), range(0, len(set(data['bodyType'])))))
       data['body_id'] = data['bodyType'].map(body_label)
       data=data.drop(['bodyType'], axis=1)

    model_label = dict(zip(sorted(list(set(data['model']))), range(0, len(set(data['model'])))))
    data['model_id'] = data['model'].map(model_label)

    data = data.drop(['regYear', 'regMonth', 'model'],axis=1)

    data['month_id'] = data['date'].apply(lambda x: x.month)
    data['sales_year'] = data['date'].apply(lambda x: x.year)
    data['time_id'] = list(map(lambda x, y: (x-2016)*12+y, data['sales_year'], data['month_id']))
    data = data.drop(['date'], axis=1)
    data = data.rename(columns={'salesVolume': 'label'})

    return data

'*****************************************预处理所有文件*************************************************'
sales_data = prepare(sales_data)
print(sales_data.head())
search_data = prepare(search_data)
# print(search_data.tail())
# 将文件拼接到数据集中并补全bodytype
pivot = pd.pivot_table(sales_data, index=['model_id', 'body_id'])
pivot = pd.DataFrame(pivot).reset_index()[['model_id', 'body_id']]
final_data = pd.merge(sales_data, pivot, on='model_id', how='left')
final_data = pd.merge(final_data, search_data, how='left', on=['pro_id', 'model_id', 'sales_year', 'month_id', 'time_id'])
print(final_data.head())
# input_data = pd.concat([input_data,final_data])
#
# input_data['salesVolume'] = input_data['label']