import time, datetime, pytz
from sqlalchemy import create_engine, text
from clickhouse_driver import Client
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from engine.alpha import Alpha158
from utils.time_func import timeit
from utils.database import *
from utils.get_sql import get_sql
from engine.expr import *
from clickhouse_driver import Client
import statsmodels.api as sm
import dolphindb as ddb
import dolphindb.settings as keys
import warnings
warnings.filterwarnings("ignore")


def remove_ST(df):
    '''
    去除ST股票
    '''

    TRD_Co = get_sql(level0_csmar, 'TRD_Co')[['Stkcd', 'Stknme']]
    TRD_Co['Stkcd'] = TRD_Co['Stkcd'].apply(lambda x: f"{x:06}")
    ST = TRD_Co[TRD_Co['Stknme'].str.contains('ST')]['Stkcd']
    df['code_s'] = df['code'].str[:6]
    df = df[~df['code_s'].isin(ST)].drop('code_s', axis=1)

    print(f"[+] remove ST at {time.strftime('%c')}")

    return df


def winsorize(df, std=3):
    '''
    去极值
    '''

    column_list = df.columns.drop(['code', 'date'])
    means = df[column_list].mean()
    std_devs = df[column_list].std()
    upper_edges = means + std * std_devs
    lower_edges = means - std * std_devs
    df[column_list] = df[column_list].clip(lower=lower_edges, upper=upper_edges, axis=1)

    print(f"[+] remove extreme at {time.strftime('%c')}")

    return df


def neutralization(df):
    '''
    行业中值化
    '''

    column_list = df.columns.drop(['code','date'])

    Stock_SWL1_Clsf = get_sql(level0_joinquant, 'Stock_SWL1_Clsf')[['Stkcd', 'Indnme']].rename(columns={'Stkcd': 'code', 'Indnme': 'industry'})
    Mkt_Value = get_sql(level1_csmar, 'TRD_Dalyr')[['Stkcd', 'Date', 'MKTValue']].rename(columns={'Stkcd': 'code', 'Date': 'date'})
    Mkt_Value['date'] = Mkt_Value['date'].dt.date
    Mkt_Value = Mkt_Value[(Mkt_Value['date'] >= df['date'].min()) & (Mkt_Value['date'] <= df['date'].max())]

    df['code_s'] = df['code']
    df['code'] = df['code'].str[:6]
    Stock_SWL1_Clsf['code'] = Stock_SWL1_Clsf['code'].apply(lambda x: f"{x:06}")
    Mkt_Value['code'] = Mkt_Value['code'].apply(lambda x: f"{x:06}")

    df = pd.merge(df, Stock_SWL1_Clsf, on='code', how='left')
    df = pd.merge(df, Mkt_Value, on=['code', 'date'], how='left')
    # 删除没有市值数据的样本
    df = df.dropna()

    industry_dummies = pd.get_dummies(df['industry'], drop_first=True)
    industry_dummies = industry_dummies.astype(int)
    X = sm.add_constant(pd.concat([df['MKTValue'], industry_dummies], axis=1))

    for column in column_list:
        y = df[column]
        model = sm.OLS(y, X).fit()
        df[column] = model.resid

    df['code'] = df['code_s']
    df.drop(['code_s', 'industry', 'MKTValue'], axis=1, inplace=True)

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


def get_daily_return_from_clickhouse(start_date, end_date, f1, f2):
    client = Client('10.8.3.37', user='quant', password='xxx', port='9000', database='day_level')
    query = "SELECT code, dt, close FROM stock where dt between %(dt_s)s and %(dt_e)s ORDER BY code, dt"
    params = {'dt_s': start_date, 'dt_e': end_date}
        
    result = client.execute(query, params)
    df = pd.DataFrame(result, columns=['code', 'date', 'close'])

    ret1_name = f't{f1}_ret'
    ret2_name = f't{f2}_ret'
    df.loc[:, ret1_name] = df.groupby(['code'])['close'].apply(lambda x: (x.shift(-f1) / x - 1).astype(float)).reset_index(drop=True)
    df.loc[:, ret2_name] = df.groupby(['code'])['close'].apply(lambda x: (x.shift(-f2-1) / x.shift(-1) - 1).astype(float)).reset_index(drop=True)

    return df


def fetch_data_and_preprocess(start_date='2014-01-01'):
    # 创建连接
    client = Client('10.8.3.37', user='quant', password='xxx', port='9000', database='day_level')
    timezone = pytz.timezone('Asia/Shanghai')

    columns_str = 'code, dt, open, high, low, close, adj_factor, preclose, volume, float_a_share, avg, pct_chg'
    # 查询数据
    query = f"SELECT {columns_str} FROM stock WHERE dt > '{start_date}' ORDER BY code, dt"
    result = client.execute(query)

    columns = columns_str.split(', ')
    raw = pd.DataFrame(result, columns=columns)

    # 一些feature
    raw[raw.columns[2:]] = raw[raw.columns[2:]].astype('float')
    raw['mkt_value'] = raw['close'] * raw['float_a_share']
    raw['turnover'] = raw['volume'] / raw['float_a_share']
    raw.rename(columns={'pct_chg':'ret'}, inplace=True)

    # rm和rf
    raw['weight'] = raw.groupby('dt', sort=False, group_keys=False)['mkt_value'].apply(lambda x: x / x.sum())
    raw['weight'] = raw.groupby('code', sort=False, group_keys=False)['weight'].shift()
    raw['w_ret'] = raw['weight'] * raw['ret']
    raw['rm'] = raw['dt'].map(raw.groupby('dt', sort=False, group_keys=False)['w_ret'].sum())
    raw['rf'] = 0.015 / 250

    # 调整股价
    raw['open'] = raw['open'] * raw['adj_factor']
    raw['high'] = raw['high'] * raw['adj_factor']
    raw['low'] = raw['low'] * raw['adj_factor']
    raw['close'] = raw['close'] * raw['adj_factor']
    raw['preclose'] = raw['preclose'] * raw['adj_factor']
    raw['avg'] = raw['avg'] * raw['adj_factor']

    raw.drop(columns=['weight', 'w_ret', 'adj_factor', 'float_a_share'], inplace=True)
    raw['dt'] = raw['dt'].astype('datetime64[ns]')

    return raw


def update_alpha(raw, num_of_days=1):
    # 导入alpha
    fields, names = Alpha158().get_factors()

    # 创建mysql数据库连接
    db_uri = 'mysql+pymysql://factor:xxx@10.8.3.37:33307/dailyfactor'
    engine = create_engine(db_uri)

    old_length = raw.shape[1]

    if num_of_days == -1:  # 更新全部数据
        df = raw.reset_index(drop=True)
        # 计算alpha
        for field, name in zip(fields, names):
            start = time.time()
            df[name] = eval(expr_transform(raw.columns, field))
            end = time.time()
            print(f'%time: {end-start}, ----{expr_transform(raw.columns, field)}')

        # 后处理
        df.drop(df.columns[2:old_length], axis=1, inplace=True)
        df.rename(columns={'dt': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['code'] = df['code'].astype(str)

        df = remove_ST(df)
        df = winsorize(df)
        df = neutralization(df)
        df = standardize(df)

        # 空值
        # df = df.drop(['CORR5'], axis=1).dropna()

        # 导入数据库
        s = ddb.session(protocol=keys.PROTOCOL_DDB, keepAliveTime=100000)
        s.connect("10.8.3.37", 8848, "admin", "123456")
        dbPath = "dfs://alpha158"
        if s.existsDatabase(dbPath):
            s.dropDatabase(dbPath)
        df['date'] = pd.to_datetime(df['date'])
        dates = np.array(pd.date_range(start=df['date'].min().strftime('%Y%m%d'), end=df['date'].max().strftime('%Y%m%d')), dtype="datetime64[D]")
        db = s.database(dbName='tsinghua', partitionType=keys.VALUE, partitions=dates, dbPath=dbPath)

        for i in range(0, len(df), 1000000):
            if i == 0:
                t = s.table(data=df.iloc[i: i+1000000])
                db.createPartitionedTable(table=t, tableName='alpha158_d2d', partitionColumns='date').append(t)
            else:
                s.run("tableInsert{{loadTable('{db}', `{tb})}}".format(db=dbPath,tb='alpha158_d2d'), df.iloc[i: i+1000000])

        return df
        
    else:  # 更新最新几天的数据
        # 需要的天数
        import re
        pattern = r'\d+'
        matches = []
        for field in fields:
            matches += re.findall(pattern, field)

        num_days_needed = np.array([int(i) for i in matches]).max()+1

        # 计算alpha
        df = raw.groupby('code').tail(num_days_needed).reset_index(drop=True)
        for field, name in zip(fields, names):
            start = time.time()
            df[name] = eval(expr_transform(raw.columns, field))
            end = time.time()
            print(f'%time: {end-start}, ----{expr_transform(raw.columns, field)}')

        # 保留最新几天
        latest_days = df['dt'].sort_values().unique()[-num_of_days:]
        df_for_update = df[df['dt'].isin(latest_days)]

        # 后处理
        df_for_update.drop(df_for_update.columns[2:old_length], axis=1, inplace=True)
        df_for_update.rename(columns={'dt': 'date'}, inplace=True)
        df_for_update['date'] = pd.to_datetime(df_for_update['date']).dt.date
        df_for_update['code'] = df_for_update['code'].astype(str)

        df_for_update = remove_ST(df_for_update)
        df_for_update = winsorize(df_for_update)
        df_for_update = neutralization(df_for_update)
        df_for_update = standardize(df_for_update)
        
        # 空值
        # df_for_update = df_for_update.drop(['CORR5'], axis=1).dropna()

        # 导入数据库
        s = ddb.session(protocol=keys.PROTOCOL_DDB)
        s.connect("10.8.3.37", 8848, "admin", "123456")
        dbPath = "dfs://alpha158"
        df_for_update['date'] = pd.to_datetime(df_for_update['date'])
        s.run("tableInsert{{loadTable('{db}', `{tb})}}".format(db=dbPath,tb='alpha158_d2d'), df_for_update)
        s.close()

        # df_for_update.to_sql('alpha158', engine, if_exists='append')

        return df_for_update

if __name__ == "__main__":
    '''
    num_of_days: -1表示更新全部数据，其他数字表示更新最新几天的数据
    '''
    start_date = '2023-12-01'
    num_of_days = -1

    data = fetch_data_and_preprocess(start_date=start_date)
    a = update_alpha(data, num_of_days=num_of_days)
