import pandas as pd
import numpy as np
import json
from data import StockData
from factors import B_feature
from multiprocessing import Pool


class Analysis(B_feature):

    def __init__(self, 
                 StockData: StockData, 
                 pool_num: int = 2,
                 func_path: str = '/code/minute_factors/report_m2d/config/feature_methods.json') -> None:
        
        super().__init__()
        self.StockData = StockData
        self.func_config, self.direction = self.get_func(func_path)
        self.pool_num = pool_num
        
            
    def get_func(self, func_path):
        with open(func_path, 'r') as config_file:
            config_data = json.load(config_file)
            func_config = {
                item['name']: {
                    'method': getattr(self, item['function']),
                    'params': item.get('params', [])
                }
                for item in config_data['methods']
            }
            direction = {item['name']: item['direction'] for item in config_data['methods']}
        return func_config, direction


    def loop_cal_factors(self, df):

        results = []
        for name, config in self.func_config.items():

            method = config['method']
            params = config['params']
            result = method(df, *params) if params else method(df)
            results.append(result)

            # print(f"[+] finish {df['code'].iloc[0]} {df['date'].iloc[0]} {name}")
 
        # print(f"[+] finish {df['code'].iloc[0]} {df['date'].iloc[0]}")

        return pd.DataFrame([results], index=pd.MultiIndex.from_tuples([(df['code'].iloc[0], df['date'].iloc[0])], names=['code', 'date']), columns=self.func_config.keys())
    

    def cal_feature(self):
        print("[+] start calculating factors...")
        with Pool(processes=self.pool_num) as pool:
            result_list = pool.map(self.loop_cal_factors, [group for _, group in self.StockData.data.groupby(['code', 'date'])])
        print("[+] finish calculating factors")
        return pd.concat(result_list, axis=0).reset_index()
    

    def cal_ic(self, feature_df):

        feature_df = feature_df.dropna()
        return_df = self.StockData.pat_ret.dropna()
        merge_df = pd.merge(feature_df, return_df, on=['code', 'date'], how='inner')

        column_list = feature_df.columns.drop(['code','date'])
    
        ic_df = pd.DataFrame()
        rankic_df = pd.DataFrame()

        for current_date in merge_df['date'].unique():

            factors = merge_df[merge_df['date'] == current_date][column_list]
            ret = merge_df[merge_df['date'] == current_date]['pat_ret']
            ret_rank = ret.rank(method='dense')

            ic_values = pd.DataFrame([current_date.strftime("%Y-%m-%d")], columns=['date'])
            rankic_values = pd.DataFrame([current_date.strftime("%Y-%m-%d")], columns=['date'])
            for column in column_list:
                if self.direction[column] == 1:
                    factor_rank = factors[column].rank(method='dense')
                else:
                    factor_rank = factors[column].rank(method='dense', ascending=False)
                # IC因子与未来收益率相关系数（未考虑方向），RankIC因子排名与未来收益率排名相关系数
                ic_values[column] = np.corrcoef(factors[column], ret)[0, 1]
                rankic_values[column] = np.corrcoef(factor_rank, ret_rank)[0, 1]

            ic_df = pd.concat([ic_df, ic_values], axis=0, ignore_index=True)
            rankic_df = pd.concat([rankic_df, rankic_values], axis=0, ignore_index=True)
        
        rankic_mean = rankic_df.iloc[:, 1:].mean()
        rankic_std = rankic_df.iloc[:, 1:].std()

        ic_ir = ic_df.iloc[:, 1:].mean() / ic_df.iloc[:, 1:].std()
        ic_over0 = (ic_df.iloc[:, 1:] > 0).sum() / len(ic_df)

        result = pd.concat([rankic_mean, rankic_std, ic_ir, ic_over0], axis=1)
        result.columns = ['rankic_mean', 'rankic_std', 'ic_ir', 'ic_over0']
        
        return result