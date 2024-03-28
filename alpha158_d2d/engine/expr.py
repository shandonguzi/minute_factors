import numpy as np
import pandas as pd
from numpy.linalg import inv


def idxmax(series, N):
    series = series.rolling(N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
    return series


def idxmin(series, N):
    series = series.rolling(N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
    return series


def quantile(se, N, qscore):
    return se.rolling(N, min_periods=1).quantile(qscore)


def greater(left, right):
    return np_pair('maximum', left, right)


def less(left, right):
    return np_pair('minimum', left, right)


def np_pair(func, series_left, series_right):
    return getattr(np, func)(series_left, series_right)


def Abs(se):
    return np.abs(se)


def log(se):
    return np.log(se)


# def rolling(se, N, func):
#     se = pd.Series(se)
#     ind = getattr(se.rolling(window=N), func)()
#     return ind


def shift(se, N):
    return pd.Series(se).shift(N)


def mean(se, N):
    return se.rolling(N).mean()
    # return rolling(se, N, 'mean')


def sum(se, N):
    return se.rolling(N).sum()
    # return rolling(se, N, 'sum')


def max(se, N):
    return se.rolling(N).max()
    # return rolling(se, N, 'max')


def min(se, N):
    return se.rolling(N).min()
    # return rolling(se, N, 'min')


def max_col(df):
    return df.max(axis=1)


def prod(se, N):
    return se.rolling(N).prod()
    # return rolling(se, N, 'prod')


def std(se, N):
    return se.rolling(N).std()
    # return rolling(se, N, 'std')


def skew(se, N):
    return se.rolling(N).skew()
    # return rolling(se, N, 'skew')


def z_score(se, N):
    se = se.values
    return pd.Series([np.nan] * (N-1) + [(se[i:i+N][-1]-se[i:i+N].mean()) / se[i:i+N].std() for i in range(se.shape[0]-N+1)])


def corr(se_left, se_right, N):
    se_left = se_left.replace(np.inf, np.nan)
    se_right = se_right.replace(np.inf, np.nan)
    return se_left.rolling(N, min_periods=int(N*0.5)).corr(se_right)


def co_skewness(se_left, se_right):
    return (se_left*se_right**2).mean() / (se_left**2).mean()**0.5 / (se_right**2).mean()


def top(se, N1, N2, output='mean'):
    se = se.values
    return [np.nan] * (N2-1) + [np.sort(se[i:i+N2])[:N1].mean() for i in range(se.shape[0]-N2+1)]


def residual(y, X, N, output=None):
    y = y.values
    X = np.concatenate([np.ones((y.shape[0], 1)), X], axis=1)

    result = []
    for i in range(y.shape[0]-N+1):

        y_seg = y[i:i+N]
        X_seg = X[i:i+N]
        try:
            beta = inv(X_seg.T@X_seg)@X_seg.T@y_seg
        except:
            print(f'singular matrix, row num: {N}')
            result.append(np.nan)
            continue

        res = y_seg-X_seg@beta

        if output=='std':
            result.append(res.std())
        elif output=='skew':
            result.append(np.mean((res - res.mean()) ** 3))
        elif output=='co_skewness':
            result.append(co_skewness(res, X_seg[:,1]-X_seg[:,1].mean()))
        elif output=='cumulate':
            result.append(np.log(1+res).sum())
        elif output=='r2':
            result.append((beta.T@X_seg.T@y_seg-N*y_seg.mean()**2) / (y_seg.T@y_seg-N*y_seg.mean()**2))
        else:
            result.append(np.array(res))

    return pd.Series([np.nan]*(N-1) + result)


def beta(y, X, N):
    y = y.values
    X = np.concatenate([np.ones((y.shape[0], 1)), X], axis=1)

    result = []
    for i in range(y.shape[0]-N+1):

        y_seg = y[i:i+N]
        X_seg = X[i:i+N]
        
        try:
            beta = inv(X_seg.T@X_seg)@X_seg.T@y_seg
        except:
            print(f'singular matrix, row num: {N}')
            result.append(np.nan)
            continue

        result.append(beta[1:].sum())

    return pd.Series([np.nan]*(N-1) + result)

    # return pd.Series([np.nan]*(N-1) + [(inv(X[i:i+N].T@X[i:i+N])@X[i:i+N].T@y[i:i+N])[1:].sum() for i in range(y.shape[0]-N+1)])


def ts_groupby(df, func, col=None):
    if col:
        return df.groupby('code', sort=False, group_keys=False)[col].transform(func)
    else:
        return df.groupby('code', sort=False, group_keys=False).apply(func)
    

def cs_groupby(df, func, col=None):
    if col:
        return df.groupby('dt', sort=False, group_keys=False)[col].transform(func)
    else:
        return df.groupby('dt', sort=False, group_keys=False).apply(func)
    

def normalize(se):
    return (se - se.mean()) / se.std()


def cs_residual(y, X):
    y = y.values
    X = np.concatenate([np.ones((y.shape[0], 1)), X], axis=1)

    res = inv(X.T@X)@X.T@y
    res = y-X@res

    return pd.Series(res)


def downside_beta(ri, rm, N):
    ri = ri.values
    rm = rm.values

    ls = []
    for i in range(ri.shape[0]-N+1):
        ri_seg= ri[i:i+N]
        rm_seg= rm[i:i+N]
        idx = np.where(rm_seg<rm_seg.mean())
        ri_seg = ri_seg[idx]
        rm_seg = rm_seg[idx]
        covs = np.cov(ri_seg, rm_seg)
        ls.append(covs[0][1]/covs[1][1])
    return pd.Series([np.nan]*(N-1) + ls)


def tail_risk(se, N):
    se = se.values
    ls = []
    for i in range(se.shape[0]-N+1):
        tmp = se[i:i+N]
        mu = np.quantile(tmp, 0.05)
        ls.append(np.log(tmp[tmp<mu]/mu).mean())
    return pd.Series([np.nan]*(N-1) + ls)



def smart_Money(df, N, expo=0.5):
    # input should be df[['close', 'ret', 'volume']]
    df['s'] = (df['ret'].abs() / df['volume']**expo).fillna(0)
    df = df.values

    ls = []
    for i in range(df.shape[0]-N+1):
        df_seg = df[i:i+N]
        vwap = (df_seg[:, 2] / df_seg[:, 2].sum() * df_seg[:, 0]).sum()
        sorted_arr = df_seg[df_seg[:, -1].argsort()[::-1]]
        smart = sorted_arr[sorted_arr[:, 2].cumsum() < sorted_arr[:, 2].sum()*0.2]
        vwap_smart = (smart[:, 2] / smart[:, 2].sum() * smart[:, 0]).sum()

        ls.append(vwap_smart/vwap)
    return pd.Series([np.nan]*(N-1) + ls)



def amplitude(df, N, proportion=0.4):
    # input should be df[['close', 'high', 'low']]
    df['amp'] = df['high'] / df['low'] - 1
    df = df.values

    ls = []
    for i in range(df.shape[0]-N+1):
        df_seg = df[i:i+N]
        sorted_arr = df_seg[df_seg[:, 0].argsort()[::-1]]
        length = int(N*proportion)
        
        vhigh = sorted_arr[:length, -1].mean()
        vlow = sorted_arr[-length:, -1].mean()

        ls.append(vhigh-vlow)
    return pd.Series([np.nan]*(N-1) + ls)


def idiosyncratic_vol(date_code, y, X, N, deCorr=False, deCorr_period=6):
    df = pd.concat([date_code, y, X], axis=1)
    df['ID_vol'] = df.groupby('code', sort=False, group_keys=False).apply(lambda df: eval(expr_transform(df, f'residual(ret, {list(X.columns)}, {N}, )'.replace('\'', ''))))
    if not deCorr:
        return df['ID_vol']
    else:
        # if minute data, this will likely raise error since shifting data usually makes no changes, yielding a not full-rank matrix
        for i in range(1, deCorr_period+1):
            df[f'ID_vol_{i}'] = df.groupby('code', sort=False, group_keys=False)['ID_vol'].shift(i)
        df['ID_vol_deCorr'] = df.groupby('dt', sort=False, group_keys=False).apply(lambda df: cs_residual(df['ID_vol'], df[[f'ID_vol_{i}' for i in range(1, deCorr_period+1)]]))
        return df['ID_vol_deCorr']


def up_bottom_line(df, N1, N2):
    candle_up = df['high'] - df[['open', 'close']].max(axis=1)
    candle_up_standard = candle_up / mean(candle_up, N1)
    df['candle_up_std'] = std(candle_up_standard, N2)

    william_down = df['close'] - df['low']
    william_down_standard = william_down / mean(william_down, N1)
    df['william_down_mean'] = mean(william_down_standard, N2)

    df['candle_up_std_zscore'] = cs_groupby(df, normalize, 'candle_up_std')
    df['william_down_mean_zscore'] = cs_groupby(df, normalize, 'william_down_mean')
    
    return df['candle_up_std_zscore'] + df['william_down_mean_zscore']


def W_cut(df, N):
    '''
    minute input, daily output
    '''
    money_q16 = df.groupby(['code', 'date'], sort=False, group_keys=False)['money'].quantile(1/16)
    ret = df.groupby(['code', 'date'], sort=False, group_keys=False)['close'].last().pct_change()

    tmp = pd.concat([money_q16, ret], axis=1)

    def func(df):
        df = df.values

        ls = []
        for i in range(df.shape[0]-N+1):
            seg = df[i:i+N]
            sorted_arr = seg[seg[:, 0].argsort()[::-1]]
            length = int(N/2)

            Mhigh = sorted_arr[:length, 1].sum()
            Mlow = sorted_arr[-length:, 1].sum()

            ls.append(Mhigh-Mlow)
        return pd.Series([np.nan]*(N-1) + ls)


    return tmp.groupby('code').apply(func)


def CGO2(df, N):
    '''
    minute input, daily output
    can use daily input, but have to change code structure (groupby no longer needed). Kept minute input to cope with other features
    unlike in report, here use t data to calculate t feature. shifting happens when using this feature in prediction.
    '''
    avg_price = df.groupby(['code', 'date'], sort=False, group_keys=False)['money'].sum() / df.groupby(['code', 'date'])['volume'].sum()
    turnover = df.groupby(['code', 'date'], sort=False, group_keys=False)['turnover'].last()
    close = df.groupby(['code', 'date'], sort=False, group_keys=False)['close'].last()

    tmp = pd.concat([avg_price, turnover, close], axis=1)

    def func(df):
        df = df.values

        ls = []
        for i in range(df.shape[0]-N+1):
            seg = df[i:i+N]

            weights = np.zeros(N)
            for t in range(N):
                if t==0:
                    weights[N-t-1] = seg[-1,1]
                else:
                    weights[N-t-1] = np.prod(1 - seg[t-2:, 1]) * seg[t-1, 1]
            weights = weights / weights.sum()
            rp = (weights * seg[:, 0]).sum()

            ls.append((seg[-1,2]-rp)/rp)
        return pd.Series([np.nan]*(N-1) + ls)


    return tmp.groupby('code').apply(func)


def CGO(df, N):
    tmp = df[['code', 'avg', 'turnover', 'close']]

    def func(df):
        df = df.values

        ls = []
        for i in range(df.shape[0]-N+1):
            seg = df[i:i+N]

            weights = np.zeros(N)
            for t in range(N):
                if t==0:
                    weights[N-t-1] = seg[-1,1]
                else:
                    weights[N-t-1] = np.prod(1 - seg[t-2:, 1]) * seg[t-1, 1]
            weights = weights / weights.sum()
            rp = (weights * seg[:, 0]).sum()

            ls.append((seg[-1,2]-rp)/rp)
        return pd.Series([np.nan]*(N-1) + ls)


    return tmp.groupby('code').apply(func)


def APB(df, N):
    '''
    minute input, daily output
    '''
    vwap = df.groupby(['code', 'date'], sort=False, group_keys=False).apply(lambda x: (x['volume'] / x['volume'].sum() * x['close']).sum())
    volu = df.groupby(['code', 'date'], sort=False, group_keys=False)['volume'].sum()

    tmp = pd.concat([vwap, volu], axis=1)
    
    def func(df):
        df = df.values

        ls = []
        for i in range(df.shape[0]-N+1):
            seg = df[i:i+N]

            weights = seg[:,1] / seg[:,1].mean()
            
            ls.append(np.log(seg[:,0].mean() / (weights * seg[:,0]).sum()))

        return pd.Series([np.nan]*(N-1) + ls)


    return tmp.groupby('code').apply(func)


def team_coin(df, N, daily=False):
    '''
    daily input, daily output
    '''
    df['turnover_chg'] = df.groupby('code')['turnover'].diff()
    df['turnover_chg_csmean'] = df['dt'].map(df.groupby('dt')['turnover_chg'].mean())

    df['inter'] = df['ret']
    df['inter_mean'] = df.groupby('code')['inter'].rolling(N).mean().values
    df['inter_std'] = df.groupby('code')['inter'].rolling(N).std().values
    df['inter_std_csmean'] = df['dt'].map(df.groupby('dt')['inter_std'].mean()).values
    df['inter_mean_volreverse'] = df['inter_mean'] * np.where(df['inter_std'] < df['inter_std_csmean'], -1, 1)
    df['inter_trnreverse'] = df['inter'] * np.where(df['turnover_chg'] < df['turnover_chg_csmean'], -1, 1)
    df['inter_trnreverse_mean'] = df.groupby('code')['inter_trnreverse'].rolling(N).mean().values

    df['intra'] = df['close'] / df['open'] - 1
    df['intra_mean'] = df.groupby('code')['intra'].rolling(N).mean().values
    df['intra_std'] = df.groupby('code')['intra'].rolling(N).std().values
    df['intra_std_csmean'] = df['dt'].map(df.groupby('dt')['intra_std'].mean()).values
    df['intra_mean_volreverse'] = df['intra_mean'] * np.where(df['intra_std'] < df['intra_std_csmean'], -1, 1)
    df['intra_trnreverse'] = df['intra'] * np.where(df['turnover_chg'] < df['turnover_chg_csmean'], -1, 1)
    df['intra_trnreverse_mean'] = df.groupby('code')['intra_trnreverse'].rolling(N).mean().values

    if not daily:
        df['preclose'] = df.groupby('code')['close'].shift(N)
    df['over'] = df['open'] / df['preclose'] - 1
    df['over_mean'] = df.groupby('code')['over'].rolling(N).mean().values
    df['over_std'] = df.groupby('code')['over'].rolling(N).std().values
    df['over_std_csmean'] = df['dt'].map(df.groupby('dt')['over_std'].mean()).values
    df['over_mean_volreverse'] = df['over_mean'] * np.where(df['over_std'] < df['over_std_csmean'], -1, 1)
    df['over_trnreverse'] = df['over'] * np.where(df['turnover_chg'] < df['turnover_chg_csmean'], -1, 1)
    df['over_trnreverse_mean'] = df.groupby('code')['over_trnreverse'].rolling(N).mean().values
    
    return df[['inter_mean_volreverse', 'inter_trnreverse_mean', 'intra_mean_volreverse', 'intra_trnreverse_mean', 'over_mean_volreverse', 'over_trnreverse_mean']].mean(axis=1)


def expr_transform(cols, expr):
    expr = expr.replace('[', 'pd.concat([')
    expr = expr.replace(']', '], axis=1)')

    for col in cols:
        expr = expr.replace(col, f'df["{col}"]')
    return expr



def calc_expr(df: pd.DataFrame, expr: str):
    if expr in list(df.columns):
        return df[expr]

    expr = expr_transform(df, expr)
    try:
        se = eval(expr)
        return se
    except:
        raise NameError('语句{}——eval异常'.format(expr))
    # shift(close,1) -> shift(df['close'],1)
    return None
