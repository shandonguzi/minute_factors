from data import StockData
from cipher_expr import Analysis
# from cipher import Analysis
from bt import BackTest
from utils.preprocess import *
import warnings
from utils.datepair import generate_monthly_date_pairs_corrected
from utils.dolphin import *
import pandas as pd
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    columns = ['code', 'dt', 'open', 'high', 'low', 'close', 'volume', 'amount', 'avg']
    columns_types = {
        'code': 'str',
        'dt': 'datetime64[ns]',
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'int',
        'amount': 'int',
        'avg': 'float32'
    }
    clickhouse_tableName = 'minute'
    start_date = '2023-12-01'
    end_date = '2023-12-31'
    pattern = [
        0,  # 因子计算t0 9:31 ～ t0 15:00，收益率计算t1 9:31 ～ t1 15:00
        # 1,  # 因子计算t0 9:31 ～ t0 14:50，收益率计算t0 14:55 ～ t1 9:31
        2,  # 因子计算t-1 13:00 ～ t0 11:30，收益率计算t0 13:31 ~ t1 11:30
    ]

    factors_dbPath = ['dfs://factors', 'dfs://factors', 'dfs://factors']
    factors_tableName = ['factors_p0', 'factors_p1', 'factors_p2']
    returns_dbPath = ['dfs://returns', 'dfs://returns', 'dfs://returns']
    returns_tableName = ['returns_p0', 'returns_p1', 'returns_p2']
    prices_dbPath = ['dfs://prices', 'dfs://prices', 'dfs://prices']
    prices_tableName = ['prices_p0', 'prices_p1', 'prices_p2']

    # # demo
    # minute_data = StockData(columns, columns_types, start_date, end_date, table, pattern[1])
    # ana = Analysis(minute_data, pool_num=2)
    # factors = ana.cal_feature()
    # ic_results = ana.cal_ic(factors)
    # bt = BackTest(factors, returns=minute_data.pat_ret, returns_colunm='pat_ret', long_short=True)

    # # demo
    # alphalens_demo = pd.read_pickle('/code/alphalens_demo.pkl')
    # factors = alphalens_demo[['date', 'code', 'late_skew_ret']]
    # prices = alphalens_demo[['date', 'code', 'close']]
    # bt = BackTest(factors, prices=prices, prices_colunm='close', long_short=True)

    date_pairs = generate_monthly_date_pairs_corrected(start_date, end_date)

    for i in range(len(pattern)):

        # returns / prices
        return_all = pd.DataFrame()
        prices_all = pd.DataFrame()
        factor_all = pd.DataFrame()

        for start, end in date_pairs:
            s = time.time()
            print(f"[+] get data from {start} to {end} start at {time.strftime('%c')}")

            raw_data = StockData(columns, columns_types, start, end, clickhouse_tableName, pattern[i])

            return_all = pd.concat([return_all, raw_data.pat_ret], axis=0)
            prices_all = pd.concat([prices_all, raw_data.close], axis=0) 

            ana = Analysis(raw_data, pool_num=2)

            factors = ana.cal_feature()
            factor_all = pd.concat([factor_all, factors], axis=0)

            print(f"[+] get data from {start} to {end} end at {time.strftime('%c')}")
            

        to_dolphindb_all(factor_all, factors_dbPath[i], factors_tableName[i])
        to_dolphindb_all(return_all, returns_dbPath[i], returns_tableName[i])
        to_dolphindb_all(prices_all, prices_dbPath[i], prices_tableName[i])

        print(f"[+] upload data from {start_date} to {end_date} end at {time.strftime('%c')}")
        print(f"[=] pattern {pattern[i]} Cost {round(time.time() - s, 1)} seconds \n")

    # bt = BackTest(factor_all, returns=return_all, returns_colunm='pat_ret', long_short=True)
    
    '''
    去ST、去中值、行业中值化、标准化
    '''
    # factor_all = factor_all.dropna()
    # factor_all = remove(factor_all)
    # factor_all = winsorize(factor_all)
    # factor_all = neutralization(factor_all)
    # factor_all = standardize(factor_all)

    '''
    factors、returns、prices分别存入数据库，全量 / 增量
    '''
    # to_dolphindb_all(factor_all, factors_dbPath, factors_tableName)
    # to_dolphindb_all(return_all, returns_dbPath, returns_tableName)
    # to_dolphindb_all(prices_all, prices_dbPath, prices_tableName)

    # to_dolphindb_update(factor_all, factors_dbPath, factors_tableName)
    # to_dolphindb_update(return_all, returns_dbPath, returns_tableName)
    # to_dolphindb_update(prices_all, prices_dbPath, prices_tableName)

    '''
    factors新增一列
    '''
    # factor_before = get_data_from_dolphindb(factors_dbPath, factors_tableName, start_date, end_date, columns='*')
    # factor_new = factor_all[['code', 'date', 'vol_downVol']]
    # factor_all_new = pd.merge(factor_before, factor_new, on=['code', 'date'], how='inner')
    # to_dolphindb_all(factor_all_new, factors_dbPath, factors_tableName)


    '''
    从数据库读取数据
    '''
    # factor_demo = get_data_from_dolphindb(factors_dbPath, factors_tableName, start_date, end_date, columns='*')
    # return_demo = get_data_from_dolphindb(returns_dbPath, returns_tableName, start_date, end_date)
    # prices_demo = get_data_from_dolphindb(prices_dbPath, prices_tableName, start_date, end_date)

    '''
    分因子回测，todo所有因子一起测
    '''
    # bt = BackTest(factor_demo, prices=prices_demo, prices_colunm='close', long_short=True)
    # bt2 = BackTest(factor_demo, returns=return_demo, returns_colunm='pat_ret', long_short=True)