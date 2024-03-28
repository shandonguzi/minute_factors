from typing import List, Tuple, Dict
from clickhouse_driver import Client
import pandas as pd
import exchange_calendars as xcals


class StockData:

    def __init__(self,
                 columns: List[str],
                 columns_types: Dict,
                 start_date: str,
                 end_date: str,
                 table: str = 'minute',
                 pattern: int = 0) -> None:

        self.columns = columns
        self.columns_types = columns_types
        self.start_date = start_date
        self.end_date = end_date
        self.table = table
        self.pattern = pattern
        self.data, self.pat_ret, self.close = self.init()


    def init(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = self.load_data()
        # 以code、date进行groupby计算因子
        data = self.pat_rev(data)
        pat_ret = self.get_pat_return()
        close = self.get_close()
        return data, pat_ret, close


    def load_data(self) -> pd.DataFrame:
        columns_str = ", ".join(self.columns)
        client = Client('10.8.3.37', user='jddata_reader', password='xxx', port='9100', database='jqdata')
        query = f"SELECT {columns_str} FROM {self.table} where dt between %(dt_s)s and %(dt_e)s ORDER BY code, dt"
        params = {'dt_s': (pd.to_datetime(self.start_date + ' 09:31:00')).to_pydatetime(),
                'dt_e': (pd.to_datetime(self.end_date + ' 15:00:00')).to_pydatetime()}
        result = client.execute(query, params)
        # df = pd.DataFrame(result, columns=['code', 'dt', 'open', 'high', 'low', 'close', 'volume', 'amount', 'avg'])
        
        df = pd.DataFrame(result, columns=self.columns)

        for column in self.columns:
            df[column] = df[column].astype(self.columns_types[column])
            if self.columns_types[column] == 'float32':
                df[column] = df[column].round(2)

        df['dt'] = pd.to_datetime(df['dt'])
        df['date'] = df['dt'].dt.date

        # 分钟级收益率，仅日度，不考虑隔天
        df.loc[:, 'ret'] = df.groupby(['date', 'code'])['close'].apply(lambda x: (x / x.shift(1) - 1).astype(float)).reset_index(drop=True)
        df.drop(['date'], axis=1, inplace=True)

        print("[+] finish get data from clickhouse")

        return df


    def pat_rev(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.pattern == 0:
            result = data.copy()
            # 当天9:30-当天15:00的数据作为一个date
            result['date'] = result['dt'].dt.date
        elif self.pattern == 1:
            result = data.copy()
            # 当天9:30-当天14:50的数据作为一个date
            result = result[~((result['dt'].dt.time > pd.to_datetime('14:50').time()) & 
                    (result['dt'].dt.time <= pd.to_datetime('15:00').time()))]
            result['date'] = result['dt'].dt.date
        elif self.pattern == 2:
            result = data.copy()
            # 昨天13:01-当天11:30的数据作为一个date
            result['date'] = result['dt'].apply(lambda x: x.date() if x.time() <= pd.to_datetime('11:30').time() else x.date() + pd.Timedelta(days=1))
        
        result['date'] = pd.to_datetime(result['date'])
        print("[+] finish pattern revise")

        return result
    

    def get_pat_return(self) -> pd.DataFrame:

        '''
        pat_ret对应date时间段后的收益率
        pattern == 0: pat_ret对应次日9:31-次日15:00的收益率，即根据今日9:31-15:00因子，次日9:31建仓，次日15:00平仓
        pattern == 1: pat_ret对应今日14:55-次日9:31的收益率，即根据今日9:31-14:50因子，今日14:55建仓，次日9:31平仓
        pattern == 2: pat_ret对应今日13:01-次日11:30的收益率，即根据昨日13:01-今日11:30因子，今日13:01建仓，次日11:30平仓

        '''

        client = Client('10.8.3.37', user='jddata_reader', password='xxx', port='9100', database='jqdata')

        if self.pattern == 0:
            h1, m1, h2, m2 = 9, 31, 15, 0
        elif self.pattern == 1:
            h1, m1, h2, m2 = 9, 31, 14, 55
        elif self.pattern == 2:
            h1, m1, h2, m2 = 13, 1, 11, 30

        query = f'''
        SELECT code, dt, close 
        FROM {self.table} 
        WHERE 
            dt BETWEEN %(dt_s)s AND %(dt_e)s
            AND ((toHour(dt) = {h1} AND toMinute(dt) = {m1}) OR (toHour(dt) = {h2} AND toMinute(dt) = {m2}))
        ORDER BY code, dt
        '''

        xshg = xcals.get_calendar("XSHG")
        next = xshg.next_session(self.end_date).strftime('%Y-%m-%d') \
            if xshg.is_session(self.end_date) else xshg.next_open(self.end_date).strftime('%Y-%m-%d')

        params = {'dt_s': (pd.to_datetime(self.start_date + ' 09:31:00')).to_pydatetime(),
                'dt_e': (pd.to_datetime(next + ' 15:00:00')).to_pydatetime()}
        result = client.execute(query, params)
        df = pd.DataFrame(result, columns=['code', 'dt', 'close'])
        df['dt'] = pd.to_datetime(df['dt'])

        # 计算收益率
        # ret_name = f'pat{self.pattern}_ret'
        ret_name = 'pat_ret'

        if self.pattern == 0:
            h9 = df[df.dt.dt.hour == 9]
            h9['date'] = h9['dt'].dt.date
            h15 = df[df.dt.dt.hour == 15]
            h15['date'] = h15['dt'].dt.date
            result = pd.merge(h9, h15, on=['code', 'date'], suffixes=('_9', '_15'), how='inner')
            result[ret_name] = result['close_15'] / result['close_9'] - 1
            result = result[['code', 'date', ret_name]]
            result[ret_name] = result.groupby(['code'])[ret_name].shift(-1)#.astype('float32').round(4)
            result.dropna(inplace=True)
        elif self.pattern == 1:
            result = df.copy()
            result['date'] = result['dt'].dt.date
            result['next_day_close'] = result.groupby('code')['close'].shift(-1)
            result[ret_name] = result['next_day_close'] / result['close'] - 1
            result = result[result.dt.dt.hour == 14][['code', 'date', ret_name]].dropna()
        elif self.pattern == 2:
            result = df.copy()
            result['date'] = result['dt'].dt.date
            result['next_day_close'] = result.groupby('code')['close'].shift(-1)
            result[ret_name] = result['next_day_close'] / result['close'] - 1
            result = result[result.dt.dt.hour == 13][['code', 'date', ret_name]].dropna()

        result[ret_name] = result[ret_name].astype('float32').round(4)
        
        print("[+] finish get return")

        return result
    
    def get_close(self) -> pd.DataFrame:
        client = Client('10.8.3.37', user='quant', password='xxx', port='9000', database='day_level')
        query = "SELECT code, dt, close FROM stock where dt between %(dt_s)s and %(dt_e)s ORDER BY code, dt"
        params = {'dt_s': self.start_date, 'dt_e': self.end_date}
            
        result = client.execute(query, params)
        df = pd.DataFrame(result, columns=['code', 'date', 'close'])
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = df['close'].astype('float32').round(2)
        return df

    @property
    def n_stocks(self) -> int:
        return len(self.data['code'].unique())

    @property
    def n_days(self) -> int:
        return len(self.data['date'].unique())