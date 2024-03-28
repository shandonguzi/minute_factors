import dolphindb as ddb
import dolphindb.settings as keys
import datetime
import pandas as pd


def generate_annual_date_pairs(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    start_of_years = pd.date_range(start=start_date, end=end_date, freq='AS')
    date_pairs = []
    for date in start_of_years:
        year_start = date.strftime('%Y-%m-%d')
        year_end = date + pd.offsets.YearEnd(1)
        if date.year == start_date.year:
            year_start = start_date.strftime('%Y-%m-%d')
        if date.year == end_date.year:
            year_end = end_date
        date_pairs.append((year_start, year_end.strftime('%Y-%m-%d')))

    return date_pairs


def get_alpha158_from_dolphindb(start_date, end_date, dbPath='dfs://alpha158'):

    s = ddb.session(protocol=keys.PROTOCOL_DDB)
    s.connect("10.8.3.37", 8848, "admin", "123456")

    date_pairs = generate_annual_date_pairs(start_date, end_date)

    df = pd.DataFrame()

    for start, end in date_pairs:
    
        params = {'dt_s': pd.to_datetime(start).to_pydatetime().strftime("%Y.%m.%d %H:%M:%S"),
                'dt_e': pd.to_datetime(end).to_pydatetime().strftime("%Y.%m.%d %H:%M:%S")}
        query = f"select * from alpha158_d2d where date >= {params['dt_s']} and date <= {params['dt_e']}"
        re = s.loadTableBySQL(tableName="alpha158_d2d", dbPath=dbPath, sql=query).toDF()
        
        df = pd.concat([df, re], axis=0)
    
    return df