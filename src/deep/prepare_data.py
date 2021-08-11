# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/8/9 16:15
@summary:  数据处理
"""
import math
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


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
def merge_data():
    # 读取数据
    sales_data = pd.read_csv(r'../../data/car_sales/train_sales_data.csv')
    search_data = pd.read_csv(r'../../data/car_sales/train_search_data.csv')
    sales_data = prepare(sales_data)
    search_data = prepare(search_data)

    # 将文件拼接到数据集中并补全bodytype
    pivot = pd.pivot_table(sales_data, index=['model_id', 'body_id'])
    # pivot = pd.DataFrame(pivot).reset_index()[['model_id', 'body_id']]
    # input_data = pd.merge(sales_data, pivot, on='model_id', how='left')
    # print(input_data.tail())
    input_data = pd.merge(sales_data, search_data, how='left', on=['pro_id', 'model_id', 'sales_year', 'month_id', 'time_id'])
    # print(final_data)
    input_data['salesVolume'] = input_data['label']

    # 对销量采取平滑log处理
    lg, log = 2, 1
    input_data['label'] = list(map(lambda x: x if x == np.NAN else math.log(x + 1, lg), input_data['label']))
    input_data['salesVolume'] = list(
        map(lambda x: x if x == np.NAN else math.log(x + 1, lg), input_data['salesVolume']))
    input_data['jidu_id'] = ((input_data['month_id'] - 1) / 3 + 1).map(int)

    return input_data


'********************************************特征提取*****************************************************'
def get_stat_feature(df_, month):
    data = df_.copy()
    stat_feat = []
    start = int((month - 24) / 3) * 2
    start += int((month - 24) / 4)
    start = start - 1 if start >= 1 else start

    '历史月销量'
    for last in range(1, 17):
        tmp = data.copy()
        tmp['time_id'] = list(map(lambda x: x + last + start if x + last + start <= 36 else -1, tmp['time_id']))
        tmp = tmp[~tmp['time_id'].isin([-1])][['label', 'time_id', 'pro_id', 'model_id', 'body_id']]
        tmp = tmp.rename(columns={'label': 'last_{0}_sale'.format(last)})
        data = pd.merge(data, tmp, how='left', on=['time_id', 'pro_id', 'model_id', 'body_id'])
        if last <= 6:
            stat_feat.append('last_{0}_sale'.format(last))

    '历史月popularity'
    for last in range(1, 17):
        tmp = data.copy()
        tmp['time_id'] = list(map(lambda x: x + last + start if x + last + start <= 36 else -1, tmp['time_id']))
        tmp = tmp[~tmp['time_id'].isin([-1])][['popularity', 'time_id', 'pro_id', 'model_id', 'body_id']]
        tmp = tmp.rename(columns={'popularity': 'last_{0}_popularity'.format(last)})
        data = pd.merge(data, tmp, how='left', on=['time_id', 'pro_id', 'model_id', 'body_id'])
        if last <= 6 or (last >= 11 and last <= 13):
            stat_feat.append('last_{0}_popularity'.format(last))

    '半年销量等统计特征'
    data['1_6_sum'] = data.loc[:, 'last_1_sale':'last_6_sale'].sum(1)
    data['1_6_mea'] = data.loc[:, 'last_1_sale':'last_6_sale'].mean(1)
    data['1_6_max'] = data.loc[:, 'last_1_sale':'last_6_sale'].max(1)
    data['1_6_min'] = data.loc[:, 'last_1_sale':'last_6_sale'].min(1)
    data['jidu_1_3_sum'] = data.loc[:, 'last_1_sale':'last_3_sale'].sum(1)
    data['jidu_4_6_sum'] = data.loc[:, 'last_4_sale':'last_6_sale'].sum(1)
    data['jidu_1_3_mean'] = data.loc[:, 'last_1_sale':'last_3_sale'].mean(1)
    data['jidu_4_6_mean'] = data.loc[:, 'last_4_sale':'last_6_sale'].mean(1)
    sales_stat_feat = ['1_6_sum', '1_6_mea', '1_6_max', '1_6_min', 'jidu_1_3_sum', 'jidu_4_6_sum', 'jidu_1_3_mean',
                       'jidu_4_6_mean']
    stat_feat = stat_feat + sales_stat_feat

    'model_pro趋势特征'
    data['1_2_diff'] = data['last_1_sale'] - data['last_2_sale']
    data['1_3_diff'] = data['last_1_sale'] - data['last_3_sale']
    data['2_3_diff'] = data['last_2_sale'] - data['last_3_sale']
    data['2_4_diff'] = data['last_2_sale'] - data['last_4_sale']
    data['3_4_diff'] = data['last_3_sale'] - data['last_4_sale']
    data['3_5_diff'] = data['last_3_sale'] - data['last_5_sale']
    data['jidu_1_2_diff'] = data['jidu_1_3_sum'] - data['jidu_4_6_sum']
    trend_stat_feat = ['1_2_diff', '1_3_diff', '2_3_diff', '2_4_diff', '3_4_diff', '3_5_diff', 'jidu_1_2_diff']
    stat_feat = stat_feat + trend_stat_feat

    '是否是沿海城市'
    yanhaicity = {1, 2, 5, 7, 9, 13, 16, 17}
    data['is_yanhai'] = list(map(lambda x: 1 if x in yanhaicity else 0, data['pro_id']))

    '春节月'
    data['is_chunjie'] = list(map(lambda x: 1 if x == 2 or x == 13 or x == 26 else 0, data['time_id']))
    data['is_chunjie_before'] = list(map(lambda x: 1 if x == 1 or x == 12 or x == 25 else 0, data['time_id']))
    data['is_chunjie_late'] = list(map(lambda x: 1 if x == 3 or x == 14 or x == 27 else 0, data['time_id']))
    month_city_stat_feat = ['is_chunjie', 'is_chunjie_before', 'is_chunjie_late', 'is_yanhai']
    stat_feat = stat_feat + month_city_stat_feat

    '两个月销量差值'
    'model 前两个月的销量差值'
    pivot = pd.pivot_table(data, index=['model_id'], values='1_2_diff', aggfunc=np.sum)
    pivot = pd.DataFrame(pivot).rename(columns={'1_2_diff': 'model_1_2_diff_sum'}).reset_index()
    data = pd.merge(data, pivot, on=['model_id'], how='left')
    'pro 前两个月的销量差值'
    pivot = pd.pivot_table(data, index=['pro_id'], values='1_2_diff', aggfunc=np.sum)
    pivot = pd.DataFrame(pivot).rename(columns={'1_2_diff': 'pro_1_2_diff_sum'}).reset_index()
    data = pd.merge(data, pivot, on=['pro_id'], how='left')
    'model,pro 前两个月的销量差值'
    pivot = pd.pivot_table(data, index=['pro_id', 'model_id'], values='1_2_diff', aggfunc=np.sum)
    pivot = pd.DataFrame(pivot).rename(columns={'1_2_diff': 'model_pro_1_2_diff_sum'}).reset_index()
    data = pd.merge(data, pivot, on=['pro_id', 'model_id'], how='left')
    pivot = pd.pivot_table(data, index=['pro_id', 'model_id'], values='1_2_diff', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'1_2_diff': 'model_pro_1_2_diff_mean'}).reset_index()
    data = pd.merge(data, pivot, on=['pro_id', 'model_id'], how='left')
    two_month_stat_feat = ['model_1_2_diff_sum', 'pro_1_2_diff_sum', 'model_pro_1_2_diff_sum',
                           'model_pro_1_2_diff_mean']
    stat_feat = stat_feat + two_month_stat_feat

    '月份'
    count_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    data['count_month'] = list(map(lambda x: count_month[int(x - 1)], data['month_id']))
    jiaqibiao = [[11, 12, 8, 10, 10, 9, 10, 8, 9, 13, 8, 9], [12, 9, 8, 11, 10, 8, 10, 8, 8, 14, 8, 10], [9, 11, 9, 11]]
    data['count_jiaqi'] = list(
        map(lambda x, y: jiaqibiao[int(x - 2016)][int(y - 1)], data['sales_year'], data['month_id']))
    stat_feat.append('count_month')
    stat_feat.append('count_jiaqi')

    '环比'
    data['huanbi_1_2'] = data['last_1_sale'] / data['last_2_sale']
    data['huanbi_2_3'] = data['last_2_sale'] / data['last_3_sale']
    data['huanbi_3_4'] = data['last_3_sale'] / data['last_4_sale']
    data['huanbi_4_5'] = data['last_4_sale'] / data['last_5_sale']
    data['huanbi_5_6'] = data['last_5_sale'] / data['last_6_sale']
    ring_ratio_stat_feat = ['huanbi_1_2', 'huanbi_2_3', 'huanbi_3_4', 'huanbi_5_6']
    stat_feat = stat_feat + ring_ratio_stat_feat

    'add环比比'
    data['huanbi_1_2_2_3'] = data['huanbi_1_2'] - data['huanbi_2_3']
    data['huanbi_2_3_3_4'] = data['huanbi_2_3'] - data['huanbi_3_4']
    data['huanbi_3_4_4_5'] = data['huanbi_3_4'] - data['huanbi_4_5']
    data['huanbi_4_5_5_6'] = data['huanbi_4_5'] - data['huanbi_5_6']
    two_ring_ratio_stat_feat = ['huanbi_1_2_2_3', 'huanbi_2_3_3_4', 'huanbi_3_4_4_5', 'huanbi_4_5_5_6']
    stat_feat = stat_feat + two_ring_ratio_stat_feat

    '该月该省份bodytype销量的占比与涨幅'
    for i in range(1, 7):
        last_time = 'last_{0}_sale'.format(i)
        pivot = pd.pivot_table(data, index=['time_id', 'pro_id', 'body_id'], values=last_time, aggfunc=np.sum)
        pivot = pd.DataFrame(pivot).rename(columns={last_time: 'pro_body_last_{0}_sale_sum'.format(i)}).reset_index()
        data = pd.merge(data, pivot, on=['time_id', 'pro_id', 'body_id'], how='left')
        data['last_{0}_sale_ratio_pro_body_last_{0}_sale_sum'.format(i, i)] = list(
            map(lambda x, y: x / y if y != 0 else 0, data[last_time], data['pro_body_last_{0}_sale_sum'.format(i)]))
        stat_feat.append('last_{0}_sale_ratio_pro_body_last_{0}_sale_sum'.format(i, i))
        if i >= 2:
            data['last_{0}_{1}_sale_pro_body_diff'.format(i - 1, i)] = data[
                                                                           'last_{0}_sale_ratio_pro_body_last_{0}_sale_sum'.format(
                                                                               i - 1)] - data[
                                                                           'last_{0}_sale_ratio_pro_body_last_{0}_sale_sum'.format(
                                                                               i)]
            stat_feat.append('last_{0}_{1}_sale_pro_body_diff'.format(i - 1, i))

    '该月该省份总销量占比与涨幅'
    for i in range(1, 7):
        last_time = 'last_{0}_sale'.format(i)
        pivot = pd.pivot_table(data, index=['time_id', 'pro_id'], values=last_time, aggfunc=np.sum)
        pivot = pd.DataFrame(pivot).rename(columns={last_time: 'pro__last_{0}_sale_sum'.format(i)}).reset_index()
        data = pd.merge(data, pivot, on=['time_id', 'pro_id'], how='left')
        data['last_{0}_sale_ratio_pro_last_{0}_sale_sum'.format(i, i)] = list(
            map(lambda x, y: x / y if y != 0 else 0, data[last_time], data['pro__last_{0}_sale_sum'.format(i)]))
        stat_feat.append('last_{0}_sale_ratio_pro_last_{0}_sale_sum'.format(i, i))
        if i >= 2:
            data['model_last_{0}_{1}_sale_pro_diff'.format(i - 1, i)] = data[
                                                                            'last_{0}_sale_ratio_pro_last_{0}_sale_sum'.format(
                                                                                i - 1)] - data[
                                                                            'last_{0}_sale_ratio_pro_last_{0}_sale_sum'.format(
                                                                                i)]
            stat_feat.append('model_last_{0}_{1}_sale_pro_diff'.format(i - 1, i))

    'popularity的涨幅占比'
    data['huanbi_1_2popularity'] = (data['last_1_popularity'] - data['last_2_popularity']) / data['last_2_popularity']
    data['huanbi_2_3popularity'] = (data['last_2_popularity'] - data['last_3_popularity']) / data['last_3_popularity']
    data['huanbi_3_4popularity'] = (data['last_3_popularity'] - data['last_4_popularity']) / data['last_4_popularity']
    data['huanbi_4_5popularity'] = (data['last_4_popularity'] - data['last_5_popularity']) / data['last_5_popularity']
    data['huanbi_5_6popularity'] = (data['last_5_popularity'] - data['last_6_popularity']) / data['last_6_popularity']
    popularity_ratio_stat_feat = ['huanbi_1_2popularity', 'huanbi_2_3popularity', 'huanbi_3_4popularity',
                                  'huanbi_4_5popularity', 'huanbi_5_6popularity']
    stat_feat = stat_feat + popularity_ratio_stat_feat

    'popu_modelpopularity'
    for i in range(1, 7):
        last_time = 'last_{0}_popularity'.format(i)
        pivot = pd.pivot_table(data, index=['time_id', 'model_id'], values=last_time, aggfunc=np.sum)
        pivot = pd.DataFrame(pivot).rename(
            columns={last_time: 'model__last_{0}_popularity_sum'.format(i)}).reset_index()
        data = pd.merge(data, pivot, on=['time_id', 'model_id'], how='left')
        data['last_{0}_popularity_ratio_model_last_{0}_popularity_sum'.format(i, i)] = list(
            map(lambda x, y: x / y if y != 0 else 0, data[last_time], data['model__last_{0}_popularity_sum'.format(i)]))
        stat_feat.append('last_{0}_popularity_ratio_model_last_{0}_popularity_sum'.format(i, i))

    'body month 增长率popularitydemo4'
    for i in range(1, 7):
        last_time = 'last_{0}_popularity'.format(i)
        pivot = pd.pivot_table(data, index=['time_id', 'body_id'], values=last_time, aggfunc=np.sum)
        pivot = pd.DataFrame(pivot).rename(columns={last_time: 'body_last_{0}_popularity_sum'.format(i)}).reset_index()
        data = pd.merge(data, pivot, on=['time_id', 'body_id'], how='left')
        data['last_{0}_popularity_ratio_body_last_{0}_popularity_sum'.format(i, i)] = list(
            map(lambda x, y: x / y if y != 0 else 0, data[last_time], data['body_last_{0}_popularity_sum'.format(i)]))
        if i >= 2:
            data['last_{0}_{1}_popularity_body_diff'.format(i - 1, i)] = (data[
                                                                              'last_{0}_popularity_ratio_body_last_{0}_popularity_sum'.format(
                                                                                  i - 1)] - data[
                                                                              'last_{0}_popularity_ratio_body_last_{0}_popularity_sum'.format(
                                                                                  i)]) / data[
                                                                             'last_{0}_popularity_ratio_body_last_{0}_popularity_sum'.format(
                                                                                 i)]
            stat_feat.append('last_{0}_{1}_popularity_body_diff'.format(i - 1, i))

    '同比一年前的增长'
    data["increase16_4"] = (data["last_16_sale"] - data["last_4_sale"]) / data["last_16_sale"]
    pivot = pd.pivot_table(data, index=["model_id", "time_id"], values='last_12_sale', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'last_12_sale': 'mean_province'}).reset_index()
    data = pd.merge(data, pivot, on=["model_id", "time_id"], how="left")
    pivot = pd.pivot_table(data, index=["model_id", "time_id"], values='last_12_sale', aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={'last_12_sale': 'min_province'}).reset_index()
    data = pd.merge(data, pivot, on=["model_id", "time_id"], how="left")
    '前4个月车型的同比'
    for i in range(1, 5):
        pivot = pd.pivot_table(data, index=["model_id", "time_id"], values='last_{0}_sale'.format(i), aggfunc=np.mean)
        pivot = pd.DataFrame(pivot).rename(
            columns={'last_{0}_sale'.format(i): 'mean_province_{0}'.format(i)}).reset_index()
        data = pd.merge(data, pivot, on=["model_id", "time_id"], how="left")
        pivot = pd.pivot_table(data, index=["model_id", "time_id"], values='last_{0}_sale'.format(i + 12),
                               aggfunc=np.mean)
        pivot = pd.DataFrame(pivot).rename(
            columns={'last_{0}_sale'.format(i + 12): 'mean_province_{0}'.format(i + 12)}).reset_index()
        data = pd.merge(data, pivot, on=["model_id", "time_id"], how="left")
    data["increase_mean_province_14_2"] = (data["mean_province_14"] - data["mean_province_2"]) / data[
        "mean_province_14"]
    data["increase_mean_province_13_1"] = (data["mean_province_13"] - data["mean_province_1"]) / data[
        "mean_province_13"]
    data["increase_mean_province_16_4"] = (data["mean_province_16"] - data["mean_province_4"]) / data[
        "mean_province_16"]
    data["increase_mean_province_15_3"] = (data["mean_province_15"] - data["mean_province_3"]) / data[
        "mean_province_15"]
    new_stat_feat = ["mean_province", "min_province", "increase16_4", "increase_mean_province_15_3",
                     "increase_mean_province_16_4", "increase_mean_province_14_2", "increase_mean_province_13_1"]

    return data, stat_feat + new_stat_feat

def cate_feat(df, features):
    """处理类别特征"""
    feature_max_idx = {}
    # 类id
    for feature in features:
        lbe = LabelEncoder()
        df[feature] = lbe.fit_transform(df[feature].astype(str))
        feature_max_idx[feature] = df[feature].max() + 1
    return df, feature_max_idx

def cate_embedding(df, features):
    """类特征emdedding"""
    # {'pro_id': 22, 'body_id': 4, 'model_id': 82, 'month_id': 12, 'jidu_id': 4, 'sales_year': 2}
    features_embedding = None
    len = df.shape[0]
    for i, feature in enumerate(features):
        if feature == 'pro_id':
            embed = tf.random.normal([len, 4], mean=0.0, stddev=1.0, dtype=tf.float32)
        elif feature == 'model_id':
            embed = tf.random.normal([len, 8], mean=0.0, stddev=1.0, dtype=tf.float32)
        else:
            embed = tf.random.normal([len, 2], mean=0.0, stddev=1.0, dtype=tf.float32)

        id_feature = df[feature].to_list()
        feature_batch = tf.constant([id_feature])
        embedding = tf.nn.embedding_lookup(embed, feature_batch)
        embedding = tf.reshape(embedding, [embedding.shape[1], embedding.shape[2]])
        if i == 0:
            features_embedding = embedding
        else:
            features_embedding = tf.concat([features_embedding, embedding], axis=-1)

    features_embedding = features_embedding.numpy()
    return features_embedding

# input_data = merge_data()
# df, stat_feat = get_stat_feature(input_data, 24)
# features = ['pro_id', 'body_id', 'model_id', 'month_id', 'jidu_id', 'sales_year']
# tmp = df[features]
# cate_feat_df, feature_max_idx = cate_feat(tmp, features)
#
# embed = cate_embedding(cate_feat_df, features)
# col_names = ['col' + str(i) for i in range(embed.shape[1])]
# d = pd.DataFrame(data=np.array(embed), columns=col_names)
#
# print(embed)
# print(col_names)
# print(d)
# len = cate_feat_df.shape[0]
# #
# embed = tf.random.normal([len, 8], mean=0.0, stddev=1.0, dtype=tf.float32)
# print(embed)
# pro_id_feature = cate_feat_df['pro_id'].to_list()
# feature_batch = tf.constant([pro_id_feature])
# print(feature_batch)
# get_embedding1 = tf.nn.embedding_lookup(embed, feature_batch)
# get_embedding1 = tf.reshape(get_embedding1, [get_embedding1.shape[1], get_embedding1.shape[2]])
# print(get_embedding1.numpy())
# a = tf.concat([get_embedding1, get_embedding1], axis=-1)
# print(a.shape)


df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter1', 'number1'])