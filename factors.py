import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from numba import jit
from numpy.lib.stride_tricks import as_strided as stride


'''
因子计算均基于pandas dataframe
'''


# '''
# rolling多列
# '''
# def roll(df, w, min, **kwargs):
#     v = df.values
#     d0, d1 = v.shape
#     s0, s1 = v.strides

#     a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))
#     valid_windows = [values for values in a if len(values) >= min]

#     rolled_df = pd.concat({
#         row: pd.DataFrame(values, columns=df.columns)
#         for row, values in zip(df.index[w-1:], valid_windows)
#     })

#     return rolled_df.groupby(level=0, **kwargs)


# '''
# QRS指标
# '''
# def QRS(df):
#     model = LinearRegression()
#     model.fit(X=df['low'].values.reshape(-1, 1), y=df['high'].values)
#     𝛽 = model.coef_[0]
#     R2 = model.score(X=df['low'].values.reshape(-1, 1), y=df['high'].values)
#     return 𝛽 * R2


@jit(nopython=True)
def QRS(df):
    x = df[:, 0]
    if x.shape[0] < 50:
        return np.nan
    y = df[:, 1]

    x_mean = x.mean()
    y_mean = y.mean()

    try:
        beta = (y * (x - x_mean)).sum() / ((x - x_mean)**2).sum()
        alpha = y_mean - beta * x_mean
        y_pred = alpha + beta * x
        SS_tot = ((y - y_mean)**2).sum()
        SS_res = ((y - y_pred)**2).sum()
        R_squared = 1 - (SS_res / SS_tot)
        
        return beta * R_squared
    
    except:
        return np.nan


def corr(df):
    if df[:, 0].shape[0] < 50:
        return np.nan
    return np.corrcoef(df[:, 0], df[:, 1])[0, 1]


def QRS_beta(df):
    x = df[:, 0]
    if x.shape[0] < 50:
        return np.nan
    y = df[:, 1]

    x_mean = x.mean()

    try:
        beta = (y * (x - x_mean)).sum() / ((x - x_mean)**2).sum()
        return beta
    except:
        return np.nan  


class B_feature:

    def __init__(self):
        trading_hours = [
            ("09:30", "10:00"), ("10:00", "10:30"), 
            ("10:30", "11:00"), ("11:00", "11:30"), 
            ("13:00", "13:30"), ("13:30", "14:00"), 
            ("14:00", "14:30"), ("14:30", "15:00")
            ]
        self.half_hour_dict = {n: {"start": pd.to_datetime(start).time(), "end": pd.to_datetime(end).time()} for n, (start, end) in enumerate(trading_hours, 1)}

    '''
    report: 华泰证券——基于全频段量价特征的选股模型
    '''

    def late_skew_ret(self, data, tail_period=30):
        # 尾盘收益率偏度(默认尾盘是最后30分钟)
        late_skew_ret = skew(data.iloc[-tail_period:]['ret'], nan_policy='omit')
        return late_skew_ret

    def down_vol_perc(self, data):
        # 下行收益率波动占比(默认下行收益率为小于均值的收益率，也可改为小于0的收益率)
        mean_ret = data['ret'].mean()
        downside_volatility = data[data['ret'] < mean_ret]['ret'].std()
        total_volatility = data['ret'].std()
        down_vol_perc = downside_volatility / total_volatility
        return down_vol_perc


    def corr_ret_lastret(self, data):
        # 前后两分钟收益率的相关性
        corr_ret_lastret = data['ret'].corr(data['ret'].shift(1))
        return corr_ret_lastret


    def corr_close_nextopen(self, data):
        # 前一分钟收盘价与后一分钟开盘价的相关性
        corr_close_nextopen = data['close'].shift(1).corr(data['open'].shift(-1))
        return corr_close_nextopen


    def volume_perc_n(self, data, n_half_hour):
        # 第n个半小时成交量占全天成交量比例
        start_time = self.half_hour_dict[n_half_hour]["start"]
        end_time = self.half_hour_dict[n_half_hour]["end"]
        volume_perc_n = data[(data['dt'].dt.time >= start_time) & (data['dt'].dt.time < end_time)]['volume'].sum() / data['volume'].sum()
        return volume_perc_n


    def early_corr_volume_ret(self, data):
        # 早盘成交量与收益率的相关性(早盘默认为开盘到10点)
        early_data = data[data['dt'].dt.time <= pd.to_datetime('10:00').time()]
        early_corr_volume_ret = early_data['volume'].corr(early_data['ret'])
        return early_corr_volume_ret


    def corr_volume_amplitude(self, data):
        # 成交量与振幅的相关性
        corr_volume_amplitude = ((data['high'] - data['low']) / data['open'] * 100).corr(data['volume'])
        return corr_volume_amplitude
    
    '''
    report: 中金——量化多因子系列(12)
    '''

    def mmt_pm(self, data):
        # 下午盘动量
        return (data['close'].iloc[-1] - data['close'].iloc[-120]) / data['close'].iloc[-120]
    

    def mmt_am(self, data):
        # 上午盘动量
        return (data['close'].iloc[120] - data['close'].iloc[0]) / data['close'].iloc[0]
    

    def mmt_last30(self, data):
        # 尾盘30分钟动量
        return (data['close'].iloc[-1] - data['close'].iloc[-30]) / data['close'].iloc[-30]
    

    def mmt_paratio(self, data):
        # 上下午盘动量差
        return self.mmt_pm(data) - self.mmt_am(data)
    

    def mmt_between(self, data):
        # 去头尾动量
        return (data['close'].iloc[-30] - data['close'].iloc[30]) / data['close'].iloc[30]
    

    def mmt_ols_qrs(self, data, window_size=50):

        '''
        分钟QRS指标(过去50根k线，这样会得到190个值，但应该只有一个值，存疑，此处求QRS的均值)
        QRS指标参考——20210121-量化择时系列（1）：金融工程视角下的技术择时艺术-刘均伟

        '''
        # 自定义roll，速度太慢
        # return data.pipe(roll, w=window_size, min=window_size).apply(lambda x: QRS(x)).mean()

        # numba速度快，try防止涨停跌停股票导致的报错
        return data[['low', 'high']].rolling(window_size, method='table').apply(QRS, raw=True, engine='numba').iloc[:, 0].mean()
    

    def mmt_ols_corr_square_mean(self, data, window_size=50):
        # 分钟QRS衍生回归R2
        # return data.pipe(roll, w=window_size, min=window_size).apply(lambda x: x['high'].corr(x['low'])).pow(2).mean()
        return data[['low', 'high']].rolling(window_size, method='table').apply(corr, raw=True, engine='numba').iloc[:, 0].pow(2).mean()
    

    def mmt_ols_corr_mean(self, data, window_size=50):
        # 分钟QRS衍生相关系数均值
        # return data.pipe(roll, w=window_size, min=window_size).apply(lambda x: x['high'].corr(x['low'])).mean()
        return data[['low', 'high']].rolling(window_size, method='table').apply(corr, raw=True, engine='numba').iloc[:, 0].mean()
    

    def mmt_ols_beta_mean(self, data, window_size=50):
        # 分钟QRS衍生beta均值
        # return data.pipe(roll, w=window_size, min=window_size).apply(lambda x: LinearRegression().fit(X=x['low'].values.reshape(-1, 1), y=x['high'].values).coef_[0]).mean()
        return data[['low', 'high']].rolling(window_size, method='table').apply(QRS_beta, raw=True, engine='numba').iloc[:, 0].mean()
    

    def mmt_ols_beta_zscore_last(self, data, window_size=50):
        # 分钟QRS衍生beta标准分
        # 𝛽_list = data.pipe(roll, w=window_size, min=window_size).apply(lambda x: LinearRegression().fit(X=x['low'].values.reshape(-1, 1), y=x['high'].values).coef_[0])
        # return (𝛽_list[-1] - 𝛽_list.mean()) / 𝛽_list.std()
        qrs_beta = data[['low', 'high']].rolling(window_size, method='table').apply(QRS_beta, raw=True, engine='numba').iloc[:, 0]
        return (qrs_beta.iloc[-1] - qrs_beta.mean()) / qrs_beta.std()
    

    def mmt_topnVolumeRet(self, data, window_size):
        # n顶量成交动量
        top_n = data[data['volume'] >= data['volume'].nlargest(window_size).min()]
        return (top_n['close'].iloc[-1] - top_n['close'].iloc[0]) / top_n['close'].iloc[0]
    

    def mmt_bottomnVolumeRet(self, data, window_size):
        # n底量成交动量
        bottom_n = data[data['volume'] <= data['volume'].nsmallest(window_size).max()]
        return (bottom_n['close'].iloc[-1] - bottom_n['close'].iloc[0]) / bottom_n['close'].iloc[0]


    def vol_volumelmin(self ,data):
        # 分钟成交量标准差
        return data['volume'].std()
    

    def vol_rangelmin(self ,data):
        # 分钟极比标准差
        return (data['high'] / data['low']).std()
    

    def vol_returnlmin(self ,data):
        # 分钟收益率标准差
        return data['ret'].std()
    

    def vol_upVol(self, data):
        # 上行波动率
        mean_ret = data['ret'].mean()
        up_volatility = data[data['ret'] > mean_ret]['ret'].std()
        return up_volatility
    

    def vol_upRatio(self, data):
        # 上行收益率波动占比
        mean_ret = data['ret'].mean()
        up_volatility = data[data['ret'] > mean_ret]['ret'].std()
        total_volatility = data['ret'].std()
        vol_upRatio = up_volatility / total_volatility
        return vol_upRatio
    

    def vol_downVol(self, data):
        # 下行波动率
        mean_ret = data['ret'].mean()
        down_volatility = data[data['ret'] < mean_ret]['ret'].std()
        return down_volatility
    

    # def vol_downRatio(self, data):
    #     # 下行收益率波动占比，同华泰down_vol_perc
    #     mean_ret = data['ret'].mean()
    #     down_volatility = data[data['ret'] < mean_ret]['ret'].std()
    #     total_volatility = data['ret'].std()
    #     vol_downRatio = down_volatility / total_volatility
    #     return vol_downRatio