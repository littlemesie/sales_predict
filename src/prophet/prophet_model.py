import pandas as pd
import numpy as np
from fbprophet import Prophet

df = pd.read_csv('../../data/example_wp_log_peyton_manning.csv')


df['y'] = np.log(df['y'])

playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))  # 季后赛和超级碗比赛特别日期
m = Prophet(holidays=holidays)  # 指定节假日参数，其它参数以默认值进行训练
m.fit(df)  # 对过去数据进行训练
future = m.make_future_dataframe(freq='D', periods=365)  # 建立数据预测框架，数据粒度为天，预测步长为一年
forecast = m.predict(future)
print(forecast)
m.plot(forecast).show()  # 绘制预测效果图
m.plot_components(forecast).show()  # 绘制成分趋势图
