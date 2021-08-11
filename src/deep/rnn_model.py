# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/8/9 16:09
@summary: RNN 模型
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Model, load_model
from deep.prepare_data import merge_data
from deep.prepare_data import get_stat_feature
from deep.prepare_data import cate_feat, cate_embedding
from utils.metrics import mape, smape


'***********************************创建模型************************************************************'
embed_dim = 32
dropout = 0.5
hidden_units = [128, 64]
learning_rate = 0.001
batch_size = 32
epochs = 10

def train_model(hidden_units, dropout):

    model = Sequential()
    model.add(SimpleRNN(24))
    # for unit in hidden_units:
    #     model.add(SimpleRNN(unit))
    model.add(Dropout(dropout))  # dropout层防止过拟合
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))      # 全连接层


    return model


'***********************************分月预测************************************************************'
def train():
    input_data = merge_data()
    model = train_model(hidden_units, dropout)
    model.compile(optimizer=RMSprop(), loss='mse')

    for m in [24]:
        df, stat_feat = get_stat_feature(input_data, m)
        cate_features = ['pro_id', 'body_id', 'model_id', 'month_id', 'jidu_id', 'sales_year']
        tmp = df[cate_features]
        cate_feat_df, feature_max_idx = cate_feat(tmp, cate_features)
        print(df.shape)
        embed = cate_embedding(cate_feat_df, cate_features)
        col_names = ['col' + str(i) for i in range(embed.shape[1])]
        cate_df = pd.DataFrame(data=np.array(embed), columns=col_names)
        df = pd.concat([df, cate_df], axis=1)
        print(df.shape)
        scaler = MinMaxScaler()
        # 数据集划分
        # stat_feat = stat_feat + col_names

        all_idx = df['time_id'].between(8, m - 1)
        stat_train_X = np.array(df[all_idx][stat_feat].fillna(0))
        cate_train_X = np.array(df[all_idx][col_names].fillna(0))
        train_X = scaler.fit_transform(stat_train_X)
        train_X = np.concatenate([train_X, cate_train_X], axis=1)
        train_y = np.array(df[all_idx]['label'])
        train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
        train_y = np.reshape(train_y, (train_y.shape[0], 1))
        # test
        test_idx = df['time_id'].between(m, m)
        stat_test_X = np.array(df[test_idx][stat_feat].fillna(0))
        cate_test_X = np.array(df[test_idx][col_names].fillna(0))
        test_X = scaler.fit_transform(stat_test_X)
        test_X = np.concatenate([test_X, cate_test_X], axis=1)
        test_y = np.array(df[test_idx]['label'])

        test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
        train_y = scaler.fit_transform(train_y)

        model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size)
        pred_y = model.predict(test_X)
        pred_y = scaler.inverse_transform(pred_y)
        pred_y = np.reshape(pred_y, (pred_y.shape[0]))
        print(pred_y)

        sales_volume = [np.exp(y) - 1 for y in test_y]
        forecast_volume = [np.exp(y) - 1 for y in pred_y]

        # MSE
        mse = metrics.mean_squared_error(sales_volume, forecast_volume)
        # RMSE
        rmse = np.sqrt(metrics.mean_squared_error(sales_volume, forecast_volume))
        # mape
        mape_ = mape(np.array(sales_volume), np.array(forecast_volume))
        # smape
        smape_ = smape(np.array(sales_volume), np.array(forecast_volume))

        print('mse: {:.4f}, rmse: {:.4f}, mape: {:.4f}, smape: {:.4f}'.format(mse, rmse, mape_, smape_))

        # mse: 644441569.9255, rmse: 25385.8537, mape: 123.7668, smape: 66.1443
        # mse: 941813451.4318, rmse: 30688.9793, mape: 74.3347, smape: 48.2840 加入embedding

train()