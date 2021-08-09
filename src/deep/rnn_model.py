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
from utils.metrics import mape, smape

input_data = merge_data()


'***********************************创建模型************************************************************'
embed_dim = 32
dropout = 0.5
hidden_units = [128, 64]
learning_rate = 0.001
batch_size = 128
epochs = 10

def train_model(hidden_units, dropout):

    model = Sequential()
    model.add(SimpleRNN(24))
    # for unit in hidden_units:
    #     model.add(SimpleRNN(unit))
    model.add(Dropout(dropout))  # dropout层防止过拟合
    model.add(Dense(10))
    model.add(Dense(1))      # 全连接层
    model.add(Activation('sigmoid'))  #激活层

    return model


# model.fit(X_train, Y_train, epochs=10, batch_size=4, verbose=2)

'***********************************分月预测************************************************************'

model = train_model(hidden_units, dropout)
model.compile(optimizer=RMSprop(), loss='mse')

for m in [24]:
    df, stat_feat = get_stat_feature(input_data, m)
    scaler = MinMaxScaler()
    # 数据集划分
    all_idx = df['time_id'].between(8, m - 1)
    train_X = np.array(df[all_idx][stat_feat].fillna(0))
    train_X = scaler.fit_transform(train_X)
    train_y = np.array(df[all_idx]['label'])
    print(train_X)
    test_idx = df['time_id'].between(m, m)
    test_X = np.array(df[test_idx][stat_feat].fillna(0))
    test_X = scaler.fit_transform(test_X)
    test_y = np.array(df[test_idx]['label'])

    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
    train_y = np.reshape(train_y, (train_y.shape[0], 1))
    # test_y = np.reshape(test_y, (test_y.shape[0], 1))
    # print(train_X)
    print(test_X)

    model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size)
    pred_y = model.predict(test_X)
    pred_y  = np.reshape(pred_y, (pred_y.shape[0]))
    print(pred_y)

    sales_volume = [np.exp(y) - 1 for y in test_y]
    forecast_volume = [np.exp(y) - 1 for y in pred_y]
    #
    # # MSE
    # mse = metrics.mean_squared_error(sales_volume, forecast_volume)
    # # RMSE
    # rmse = np.sqrt(metrics.mean_squared_error(sales_volume, forecast_volume))
    # # mape
    # mape_ = mape(np.array(sales_volume), np.array(forecast_volume))
    # # smape
    # smape_ = smape(np.array(sales_volume), np.array(forecast_volume))
    #
    # print('mse: {:.4f}, rmse: {:.4f}, mape: {:.4f}, smape: {:.4f}'.format(mse, rmse, mape_, smape_))

