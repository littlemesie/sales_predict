import warnings
import numpy as np
import pandas as pd
from sklearn import metrics
from utils.metrics import mape, smape
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

'*******************************************读取数据***************************************************'
train = pd.read_csv(r'../../data/car_sales/train_sales_data.csv')
# test = pd.read_csv(r'../../data/car_sales/evaluation_public.csv')

'********************************************rule训练***********************************************'
def exp_smooth(df,alpha=0.97, base=50, start=1, win_size=3, t=24):
    # 使用三次指数平滑，根据历史销量值的变化趋势预测将来销量
    # 平滑因子，两次平滑之间间隔大小，起始编号，初始值的窗口大小，平滑周期
    # 第一次指数平滑
    df[start+base-1] = 0
    for i in range(win_size):
        df[start+base-1] += df[start+i] / win_size
    for i in range(t):
        df[start+base+i] = alpha * df[start+i] + (1 - alpha) * df[start+base+i-1]
    # 第二次指数平滑
    df[start+2 * base-1] = 0
    for i in range(win_size):
        df[start+2*base-1] += df[start+base+i] / win_size
    for i in range(t):
        df[start+2*base+i] = alpha * df[start+base+i] + (1 - alpha) * df[start+2*base+i-1]
    # 第三次指数平滑
    df[start+3 * base-1] = 0
    for i in range(win_size):
        df[start+3*base-1] += df[start+2*base+i] / win_size
    for i in range(t):
        df[start+3*base+i] =  alpha * df[start+2*base+i] + (1 - alpha) * df[start+3*base+i-1]

    # 套入公式计算未来两个月的平滑值
    t1, t2, t3 = df[start+base+t-1], df[start+2*base+t-1], df[start+3*base+t-1]
    a = 3 * t1 - 3 * t2 + t3
    b = ((6 - 5 * alpha) * t1 - 2 * (5 - 4 * alpha) * t2 + (4 - 3 * alpha) * t3) * alpha / (2 * (1 - alpha) ** 2)
    c = (t1 - 2 * t2 + t3) * alpha ** 2 / (2 * (1 - alpha) ** 2)
    for m in [25, 26]:
        df[m] = a + b * (m-t) + c * (m-t) ** 2
    return df

def rule():
    # 规则
    # 对数据取对数，缩小销量之间的差距，降低极端值的影响
    train['salesVolume'] = np.log1p(train['salesVolume'])

    # 规则
    train16 = train[(train['regYear'] == 2016)][['adcode', 'model', 'regMonth', 'salesVolume']]
    train17 = train[(train['regYear'] == 2017)][['adcode', 'model', 'regMonth', 'salesVolume']]

    # train171 = train[(train['regYear'] == 2017) & (train['regMonth'] != 12)][['adcode', 'model', 'regMonth', 'salesVolume']]
    # print(train171)
    test = train[(train['regYear'] == 2017) & (train['regMonth'] == 12)][['adcode', 'model', 'regMonth', 'salesVolume']]

    # 1~3的趋势
    df16 = train16.loc[train16['regMonth'] <= 3].groupby(['adcode', "model"], as_index=False)['salesVolume']. \
        agg({"16_3_mean": 'mean'})  # 按省份和车型统计均值
    df17 = train17.loc[train17['regMonth'] <= 3].groupby(['adcode', "model"], as_index=False)['salesVolume']. \
        agg({"17_3_mean": 'mean'})
    df = pd.merge(df17, df16, on=['adcode', 'model'], how='inner')
    df['3_factor'] = df['17_3_mean'] / df['16_3_mean']  # 17年均值除以16年均值得到趋势因子

    # 4~6的趋势
    df16 = \
    train16.loc[(train16['regMonth'] >= 4) & (train16['regMonth'] <= 6)].groupby(['adcode', "model"], as_index=False)[
        'salesVolume']. \
        agg({"16_6_mean": 'mean'})  # 按省份和车型统计均值
    df17 = \
    train17.loc[(train17['regMonth'] >= 4) & (train17['regMonth'] <= 6)].groupby(['adcode', "model"], as_index=False)[
        'salesVolume']. \
        agg({"17_6_mean": 'mean'})
    df17 = df17.merge(df16, on=['adcode', 'model'], how='inner')
    df['6_factor'] = df17['17_6_mean'] / df17['16_6_mean']  # 17年均值除以16年均值得到趋势因子
    # 7~9的趋势
    df16 = \
    train16.loc[(train16['regMonth'] >= 7) & (train16['regMonth'] <= 9)].groupby(['adcode', "model"], as_index=False)[
        'salesVolume']. \
        agg({"16_9_mean": 'mean'})  # 按省份和车型统计均值
    df17 = \
    train17.loc[(train17['regMonth'] >= 7) & (train17['regMonth'] <= 9)].groupby(['adcode', "model"], as_index=False)[
        'salesVolume']. \
        agg({"17_9_mean": 'mean'})
    df17 = df17.merge(df16, on=['adcode', 'model'], how='inner')
    df['9_factor'] = df17['17_9_mean'] / df17['16_9_mean']  # 17年均值除以16年均值得到趋势因子
    # 10~12的趋势
    df16 = \
    train16.loc[(train16['regMonth'] >= 10) & (train16['regMonth'] <= 12)].groupby(['adcode', "model"], as_index=False)[
        'salesVolume']. \
        agg({"16_12_mean": 'mean'})  # 按省份和车型统计均值
    df17 = \
    train17.loc[(train17['regMonth'] >= 10) & (train17['regMonth'] <= 12)].groupby(['adcode', "model"], as_index=False)[
        'salesVolume']. \
        agg({"17_12_mean": 'mean'})
    df17 = df17.merge(df16, on=['adcode', 'model'], how='inner')
    df['12_factor'] = df17['17_12_mean'] / df17['16_12_mean']  # 17年均值除以16年均值得到趋势因子
    # 对趋势进行幂次平滑
    up_thres, down_thres, up_ratio, down_ratio = 1.2, 0.75, 0.5, 0.5
    for factor in ['3_factor', '6_factor', '9_factor', '12_factor']:
        df.loc[df[factor] > up_thres, factor] = df.loc[df[factor] > up_thres, factor].apply(lambda x: x ** up_ratio)
        df.loc[df[factor] < down_thres, factor] = df.loc[df[factor] < down_thres, factor].apply(
            lambda x: x ** down_ratio)

    # 总体趋势
    def calc_factor(x):
        L = list(x)
        L = sorted(L)
        return 0.6 * L[0] + 0.2 * L[1] + 0.1 * L[2] + 0.1 * L[3]

    df['factor'] = df[['3_factor', '6_factor', '9_factor', '12_factor']].apply(lambda x: calc_factor(x), axis=1)
    # 对整体趋势进行后处理
    df['factor'] = df['factor'].apply(lambda x: min(x, 1.25))
    df['factor'] = df['factor'].apply(lambda x: max(x, 0.75))

    # 在省份-车型作为主键的情况下，取出16年和17年的数据，共24个月
    for m in range(1, 13):
        df = pd.merge(df, train16[train16['regMonth'] == m][['adcode', 'model', 'salesVolume']], on=['adcode', 'model'],
                      how='left').rename(columns={'salesVolume': m})
        df = pd.merge(df, train17[train17['regMonth'] == m][['adcode', 'model', 'salesVolume']], on=['adcode', 'model'],
                      how='left').rename(columns={'salesVolume': 12 + m})

    df = exp_smooth(df, alpha=0.95)

    res = pd.DataFrame()
    tmp = df[['adcode', 'model']].copy()
    trend_factor = [0.985, 0.965, 0.99, 0.985]
    for i, m in enumerate([24]):
        # 以省份-车型作为主键，计算前年，去年，最近几个月的值，然后加权得到一个当前月份的预测值
        last_year_base = 0.2 * df[m - 13].values + 0.6 * df[m - 12].values + 0.2 * df[m - 11].values
        # if m == 25:
        #     last_last_year_base = 0.8 * df[m - 24] + 0.2 * df[m - 23]
        # else:
        #     last_last_year_base = 0.2 * df[m - 25] + 0.6 * df[m - 24] + 0.2 * df[m - 23]
        if m <= 26:
            near_base = 0.2 * df[m - 3] + 0.2 * df[m - 2] + 0.3 * df[m - 1] + 0.3 * df[m]
        else:
            near_base = 0.2 * df[m - 3] + 0.2 * df[m - 2] + 0.6 * df[m - 1]

        # 按照三个的大小进行加权求和
        temp = pd.DataFrame()
        temp['near_base'] = near_base
        temp['last_year_base'] = last_year_base
        # temp['last_last_year_base'] = last_last_year_base

        def calc(row):
            L = list(row)
            L = sorted(L)
            return 0.7 * L[0] + 0.3 * L[1]
            # return 0.6 * L[0] + 0.2 * L[1] + 0.2 * L[2]

        temp['base'] = temp.apply(lambda row: calc(row), axis=1)
        base = temp['base']
        tmp['forecastVolum'] = base * df['factor'] * trend_factor[i]
        df[m] = tmp['forecastVolum']

        tmp['regMonth'] = m - 24
        res = res.append(tmp, ignore_index=True)

    test = pd.merge(test[['salesVolume', 'adcode', 'model', 'regMonth']], res, how='left', on=['adcode', 'model'])

    test['salesVolume'] = np.exp(test['salesVolume']) - 1
    test.loc[test['salesVolume'] < 0, 'salesVolume'] = 0

    test['forecastVolum'] = np.exp(test['forecastVolum']) - 1
    test.loc[test['forecastVolum'] < 0, 'forecastVolum'] = 0
    # print(test)

    return test[['salesVolume', 'forecastVolum']]

sub = rule()
# MSE
mse = metrics.mean_squared_error(np.array(sub['salesVolume']), np.array(sub['forecastVolum']))
print(mse)
# RMSE
rmse = np.sqrt(metrics.mean_squared_error(np.array(sub['salesVolume']), np.array(sub['forecastVolum'])))
print(rmse)

# mape
mape = mape(np.array(sub['salesVolume']), np.array(sub['forecastVolum']))
print(mape)
# smape
smape = smape(np.array(sub['salesVolume']), np.array(sub['forecastVolum']))
print(smape)