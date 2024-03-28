import pandas as pd
import numpy as np
from scipy.stats import skew as sk
from scipy.stats import kurtosis as kurt
from numba import jit
import warnings
warnings.filterwarnings("ignore")


def skew(series):
    return sk(series, nan_policy='omit')

def kurtosis(series):
    return kurt(series, nan_policy='omit')

def idx(series, idx):
    if isinstance(idx, int):
        return series.iloc[idx]
    elif isinstance(idx, slice):
        return series.iloc[idx]
    elif isinstance(idx, tuple):
        if len(idx) == 2:
            return series.iloc[idx[0], idx[1]]
        else:
            raise ValueError("元组索引应该只有两个元素：(行选择, 列选择)")
    else:
        return series.loc[idx]

def mean(series):
    return np.mean(series)

def sum(series):
    return np.sum(series)

def pow(series, n):
    return series.pow(n)

def less(series, num):
    return series[series <= num]

def more(series, num):
    return series[series >= num]

def std(series):
    return np.std(series)

def shift(series, n):
    return series.shift(n)

def corr(series1, series2):
    return series1.corr(series2)

def zscore_last(series):
    return (series.iloc[-1] - series.mean()) / series.std()

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

def multicorr(df):
    if df[:, 0].shape[0] < 50:
        return np.nan
    return np.corrcoef(df[:, 0], df[:, 1])[0, 1]

def multiroll(series1, series2, window, func):
    return pd.concat([series1, series2], axis=1).rolling(window, method='table').apply(func, raw=True, engine='numba')

def nlmin(series, n):
    return series.nlargest(n).min()

def nsmax(series, n):
    return series.nsmallest(n).max()

def momentum(series):
    return (series.iloc[-1] - series.iloc[0]) / series.iloc[0]

def getidx(series):
    return series.index

def pct(series):
    return series.pct_change()

def shift(series, period):
    return series.shift(period)

def expr_transform(cols, expr):
    # expr = expr.replace('[', 'pd.concat([')
    # expr = expr.replace(']', '], axis=1)')

    for col in cols:
        expr = expr.replace(col, f'df["{col}"]')
    return expr

# demo = pd.DataFrame(np.random.rand(240, 6), columns=['close', 'ret', 'low', 'high', 'open', 'volume'])

# exprs = {
#     'late_skew_ret': 'skew(idx(ret, slice(-30, None)))',
#     'down_vol_perc': 'std(less(ret, mean(ret))) / std(ret)',
#     'corr_ret_lastret': 'corr(ret, shift(ret, 1))',
#     'corr_close_nextopen': 'corr(shift(close, 1), shift(open, -1))',
#     'volume_perc_2': 'sum(idx(volume, slice(2, 4))) / sum(volume)',
#     'early_corr_volume_ret': 'corr(idx(volume, slice(30)), idx(ret, slice(30)))',
#     'corr_volume_amplitude': 'corr(volume, (high-low)/open*100)',

#     'mmt_pm': '(idx(close, -1) - idx(close, -120)) / idx(close, -120)',
#     'mmt_am': '(idx(close, 120) - idx(close, 0)) / idx(close, 0)',
#     'mmt_last30': '(idx(close, -1) - idx(close, -30)) / idx(close, -30)',
#     'mmt_paratio': '(idx(close, -1) - idx(close, -120)) / idx(close, -120) - (idx(close, 120) - idx(close, 0)) / idx(close, 0)',
#     'mmt_between': '(idx(close, -30) - idx(close, 30)) / idx(close, 30)',
#     'mmt_ols_qrs': 'mean(idx(multiroll(low, high, 50, QRS), (slice(None), 0)))',
#     'mmt_ols_corr_square_mean': 'mean(pow(idx(multiroll(low, high, 50, multicorr), (slice(None), 0)), 2))',
#     'mmt_ols_corr_mean': 'mean(idx(multiroll(low, high, 50, multicorr), (slice(None), 0)))',
#     'mmt_ols_beta_mean': 'mean(idx(multiroll(low, high, 50, QRS_beta), (slice(None), 0)))',
#     'mmt_ols_beta_zscore_last': 'zscore_last(idx(multiroll(low, high, 50, QRS_beta), (slice(None), 0)))',
#     'mmt_top50VolumeRet': 'momentum(idx(close, getidx(more(volume, nlmin(volume, 50)))))',
#     'mmt_bottom50VolumeRet': 'momentum(idx(close, getidx(less(volume, nsmax(volume, 50)))))',
#     'mmt_top20VolumeRet': 'momentum(idx(close, getidx(more(volume, nlmin(volume, 20)))))',
#     'mmt_bottom20VolumeRet': 'momentum(idx(close, getidx(less(volume, nsmax(volume, 20)))))',
#     'vol_volumelmin': 'std(volume)',
#     'vol_rangelmin': 'std(high/low)',
#     'vol_returnlmin': 'std(ret)',
#     'vol_upVol': 'std(more(ret, mean(ret)))',
#     'vol_upRatio': 'std(more(ret, mean(ret))) / std(ret)',
#     'vol_downVol': 'std(less(ret, mean(ret)))',
#     'vol_downRatio': 'std(less(ret, mean(ret))) / std(ret)',
# }

# for factor, expr in exprs.items():
#     print(factor, eval(expr_transform(demo.columns, expr)))
