from utils.get_sql import *
from utils.database import *
from clickhouse_driver import Client
import time
import pandas as pd
import statsmodels.api as sm


def remove(df, ST=True, GEM=True, STAR=True, up_down=True, stop=True, IPO=True):
    '''
    去除ST股票、创业板、科创板、涨跌停、停牌、IPO<30天
    Parameters:
    -----------
    ST: bool
        是否去除ST股票

    GEM: bool
        是否去除创业板股票

    STAR: bool
        是否去除科创板股票

    up_down: bool
        是否去除涨跌停

    stop: bool
        是否去除停牌

    IPO: bool
        是否去除IPO<30天
    '''
    
    client = Client('10.8.3.37', user='quant', password='xxx', port='9000', database='day_level')
    query = "SELECT code, dt, close, status, limit, stopping, ipo_date, st FROM stock where dt between %(dt_s)s and %(dt_e)s ORDER BY code, dt"
    params = {'dt_s': df['date'].min().strftime("%Y-%m-%d"), 'dt_e': df['date'].max().strftime("%Y-%m-%d")}
    result = client.execute(query, params)
    flag = pd.DataFrame(result, columns=['code', 'date', 'close', 'status', 'limit', 'stopping', 'ipo_date', 'st'])
    flag['date'] = pd.to_datetime(flag['date'])
    flag['ipo_date'] = pd.to_datetime(flag['ipo_date'])

    df = pd.merge(df, flag, on=['code', 'date'])

    if ST:
        df = df[df['st'] == 0]
    if GEM:
        df = df[~df['code'].str.contains('^30')]
    if STAR:
        df = df[~df['code'].str.contains('^68')]
    if up_down:
        df = df[(df.close < df.limit) & (df.close > df.stopping)]
    if stop:
        df = df[df['status'] != "停牌"]
    if IPO:
        df = df[(df['date'] - df['ipo_date']).dt.days > 30]
    
    df = df.drop(['close', 'status', 'limit', 'stopping', 'ipo_date', 'st'], axis=1)

    print(f"[+] remove at {time.strftime('%c')}")

    return df


def winsorize(df, std=3):
    '''
    去极值
    Parameters:
    -----------
    std: int
        去极值标准差倍数
    '''

    column_list = df.columns.drop(['code', 'date'])
    means = df[column_list].mean()
    std_devs = df[column_list].std()
    upper_edges = means + std * std_devs
    lower_edges = means - std * std_devs
    df[column_list] = df[column_list].clip(lower=lower_edges, upper=upper_edges, axis=1)

    print(f"[+] remove extreme at {time.strftime('%c')}")

    return df


def missinghandle(df, method='fill'):
    '''
    去缺失值
    Parameters:
    -----------
    method: str
        drop: 删除缺失值
        fill: 用行业中位数填充
    '''

    if method == 'drop':
        return df.dropna()
    
    column_list = df.columns.drop(['code','date'])
    
    client = Client('10.8.3.37', user='quant', password='xxx', port='9000', database='day_level')
    query = "SELECT code, dt, sw_l1_code FROM stock where dt between %(dt_s)s and %(dt_e)s ORDER BY code, dt"
    params = {'dt_s': df['date'].min().strftime("%Y-%m-%d"), 'dt_e': df['date'].max().strftime("%Y-%m-%d")}
    result = client.execute(query, params)
    flag = pd.DataFrame(result, columns=['code', 'date', 'sw_l1_code'])
    flag['date'] = pd.to_datetime(flag['date'])
    
    df = pd.merge(df, flag, on=['code', 'date'], how='left').dropna()

    for factor in column_list:
        df[factor] = df.groupby('sw_l1_code')[factor].transform(lambda x: x.fillna(x.median()))

    df = df.drop(['sw_l1_code'], axis=1).dropna()
    print(f"[+] missinghandle at {time.strftime('%c')}")

    return df


def neutralization(df):
    '''
    行业市值中值化
    '''

    column_list = df.columns.drop(['code','date'])

    client = Client('10.8.3.37', user='quant', password='xxx', port='9000', database='day_level')
    query = "SELECT code, dt, close, sw_l1_code, float_a_share FROM stock where dt between %(dt_s)s and %(dt_e)s ORDER BY code, dt"
    params = {'dt_s': df['date'].min().strftime("%Y-%m-%d"), 'dt_e': df['date'].max().strftime("%Y-%m-%d")}
    result = client.execute(query, params)
    flag = pd.DataFrame(result, columns=['code', 'date', 'close', 'sw_l1_code', 'float_a_share'])
    flag['date'] = pd.to_datetime(flag['date'])
    flag['mktvalue'] = (flag['float_a_share'] * flag['close']).astype('float32')
    
    df = pd.merge(df, flag[['code', 'date', 'sw_l1_code', 'mktvalue']], on=['code', 'date'], how='left')

    industry_dummies = pd.get_dummies(df['sw_l1_code'], drop_first=True)
    industry_dummies = industry_dummies.astype(int)
    X = sm.add_constant(pd.concat([df['mktvalue'], industry_dummies], axis=1))

    for column in column_list:
        y = df[column]
        model = sm.OLS(y, X).fit()
        df[column] = model.resid

    df = df.drop(['sw_l1_code', 'mktvalue'], axis=1)

    print(f"[+] neutralization at {time.strftime('%c')}")
    
    return df


def standardize(df):
    '''
    Z-score标准化
    '''

    column_list = df.columns.drop(['code', 'date'])
    means = df[column_list].mean()
    std_devs = df[column_list].std()
    df[column_list] = (df[column_list] - means) / std_devs

    print(f"[+] standardize at {time.strftime('%c')}")

    return df


def get_pool(df, pool='all'):
    '''
    股票池
    Parameters:
    -----------
    pool: str
        hs300 | sz50 | zz500
    '''

    if pool == 'all':
        print(f"[+] adjust stock pool to {pool} at {time.strftime('%c')}")
        return df
    elif pool == 'hs300':
        stock_pool = get_sql(level1_csmar, 'hs300_component').rename(columns={'Date': 'date', 'Stkcd': 'code_s'})
    elif pool == 'sz50':
        stock_pool = get_sql(level1_csmar, 'sz50_component').rename(columns={'Date': 'date', 'Stkcd': 'code_s'})
    elif pool == 'zz500':
        stock_pool = get_sql(level1_csmar, 'zz500_component').rename(columns={'Date': 'date', 'Stkcd': 'code_s'})
    
    stock_pool['date'] = pd.to_datetime(stock_pool['date'])
    stock_pool['code_s'] = stock_pool['code_s'].apply(lambda x: f"{x:06}")
    stock_pool = stock_pool[['code_s', 'date']]

    df['code_s'] = df['code'].str[:6]

    df = pd.merge(df, stock_pool, on=['code_s', 'date'], how='inner').drop(['code_s'], axis=1)

    print(f"[+] adjust stock pool to {pool} at {time.strftime('%c')}")

    return df