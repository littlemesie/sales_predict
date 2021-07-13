# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/7/13 15:39
@summary:
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

ChinaBank = pd.read_csv('../../data/ChinaBank.csv', index_col='Date', parse_dates=['Date'])

# ChinaBank.index = pd.to_datetime(ChinaBank.index)
sub = ChinaBank['2014-01':'2014-06']['Close']
# print(sub)
train = sub['2014-01':'2014-03']
test = sub['2014-04':'2014-06']
# plt.figure(figsize=(10, 10))
# print(train)
# plt.plot(train)
# plt.show()

# 利用pandas 使用差分法可以使得数据更平稳，常用的方法就是一阶差分法和二阶差分法
ChinaBank['Close_diff_1'] = ChinaBank['Close'].diff(1)
ChinaBank['Close_diff_2'] = ChinaBank['Close_diff_1'].diff(1)
# fig = plt.figure(figsize=(20,6))
# ax1 = fig.add_subplot(131)
# ax1.plot(ChinaBank['Close'])
# ax2 = fig.add_subplot(132)
# ax2.plot(ChinaBank['Close_diff_1'])
# ax3 = fig.add_subplot(133)
# ax3.plot(ChinaBank['Close_diff_2'])
# plt.show()

# 根据不同的截尾和拖尾的情况，我们可以选择AR模型，也可以选择MA模型，当然也可以选择ARIMA模型
# fig = plt.figure(figsize=(12, 8))
#
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(train, lags=20, ax=ax1)
# ax1.xaxis.set_ticks_position('bottom')
# fig.tight_layout()
#
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
# ax2.xaxis.set_ticks_position('bottom')
# fig.tight_layout()
# plt.show()

# 得到p和q的最优值
# train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)
#
# print('AIC', train_results.aic_min_order)
# print('BIC', train_results.bic_min_order)

# 模型预测
model = sm.tsa.ARIMA(sub, order=(1, 0, 0))
results = model.fit()
predict_sunspots = results.predict(start=str('2014-05'), end=str('2014-06'), dynamic=False)
print(predict_sunspots)
fig, ax = plt.subplots(figsize=(12, 8))
ax = sub.plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()