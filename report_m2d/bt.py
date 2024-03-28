from al import performance as perf
from al import plotting as plot
from al import tears
from al import utils

from typing import Tuple, Union
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


class BackTest():

    '''
    因子回测框架，基于alphalens

    Parameters
    ----------
    factor_data : pd.DataFrame
        因子数据，date时间，code代码，columns为因子值，index为[date, code]

    periods : Union[int, Tuple[int]]
        收益计算的时间周期，可以为单个整数或者整数的元组，传入prices时参与计算，传入returns时不参与计算

    n_quantiles : int
        分位数数量，每个时间节点等量分为n_quantiles组
    
    prices : pd.DataFrame
        价格数据，与periods结合计算收益率，date时间，code代码，prices_colunm为列名，与returns二选一
    
    prices_colunm : str
        价格数据的列名
    
    returns : pd.DataFrame
        收益数据，直接传入收益率，默认period为1，date时间，code代码，returns_colunm为列名，与prices二选一
        ！此处存在直接传入收益率，为了处理因子计算周期为 9:31-14:50 / 13:01-11:30 的情况
    
    returns_colunm : str
        收益数据的列名
    
    long_short : bool
        是否计算多空组合收益，True会去除market影响，适合比较不同组的收益，False会计算每组的绝对收益
    
    instructors : Union[str, Tuple[str]]
        因子分析选项，可选'statistic' 'return' 'ic' 'turnover' 'all'，可选择其中一项或者多项
        传入'statistic'会计算因子统计指标
        传入'return'会计算多空组合收益率
        传入'ic'会计算信息系数
        传入'turnover'会计算换手率
        传入'all'会计算所有指标
    '''

    def __init__(self, 
                 factors: pd.DataFrame, 
                 periods: Union[int, Tuple[int]] = (1),
                 n_quantiles: int = 5,
                 prices: pd.DataFrame = pd.DataFrame(),
                 prices_colunm: str = None,
                 returns: pd.DataFrame = pd.DataFrame(),
                 returns_colunm: str = None,
                 long_short: bool = True,
                 instructors: Union[str, Tuple[str]] = 'all') -> None:
        self.factors = factors
        self.periods = periods
        self.n_quantiles = n_quantiles
        self.prices = prices
        self.prices_colunm = prices_colunm
        self.returns = returns
        self.returns_colunm = returns_colunm
        self.long_short = long_short
        self.instructors = instructors

        self.preprocess()
        self.combine_data = self.combine_data()


    def preprocess(self) -> None:
        self.factors = self.factors.set_index(['date', 'code']).dropna()
        if not self.prices.empty:
            self.prices = self.prices.pivot(index='date', columns='code', values=self.prices_colunm)
        
    
    def combine_data(self) -> pd.DataFrame:
        if self.prices.empty == self.returns.empty:
            raise ValueError('Please provide either prices or returns')
        if self.returns.empty == bool(self.returns_colunm):
            raise ValueError('Please provide returns_colunm if returns is provided')
        
        if not self.prices.empty:
            combine_data = utils.get_clean_factor_and_forward_returns(
                        factor=self.factors,
                        prices=self.prices,
                        quantiles=self.n_quantiles,
                        periods=self.periods)
            return combine_data
        
        if not self.returns.empty:
            factor_name = self.factors.columns[0]
            factor_data = self.factors.copy().rename({factor_name: 'factor'}, axis=1)
            factor_data['factor_quantile'] = utils.quantize_factor(factor_data, self.n_quantiles).values
            combine_data = pd.merge(factor_data, self.returns.set_index(['date', 'code']), left_index=True, right_index=True, how='inner')
            combine_data = combine_data[[self.returns_colunm, 'factor', 'factor_quantile']]
            combine_data.rename(columns={self.returns_colunm: '1D'}, inplace=True)
            return combine_data
        
    def overall_description(self, period) -> None:

        def ret(df, period):
            tdays = df.loc[1].shape[0]
            long = df.loc[5]
            short = df.loc[1]
            lret = (((long[period] + 1).iloc[::eval(period[:-1])].prod())**(252 / tdays) - 1) * 100
            sret = (((short[period] + 1).iloc[::eval(period[:-1])].prod())**(252 / tdays) - 1) * 100
            lsret = lret - sret
            ls_merge = pd.merge(long[period], short[period], left_index=True, right_index=True, suffixes=('_long', '_short'))
            sharp = (ls_merge['ls'].iloc[::eval(period[:-1])].mean() / ls_merge['ls'].iloc[::eval(period[:-1])].std()) * np.sqrt(252 / eval(period[:-1]))
            win = sum(ls_merge.iloc[::eval(period[:-1])]['ls'] > 0) / len(ls_merge.iloc[::eval(period[:-1])]['ls'])
            return pd.DataFrame({'lret': lret, 'sret': sret, 'lsret': lsret, 'sharp': sharp, 'win': win, 'tdays': tdays}, index=[0])

        mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
            self.combine_data,
            by_date=True,
            by_group=False
        )

        mean_quant_ret_bydate['year'] = mean_quant_ret_bydate.index.get_level_values('date').year
        df_ret = mean_quant_ret_bydate.groupby(['year']).apply(lambda x: ret(x, period)).droplevel(1)
        
        def turnover(df, period):
            period = eval(period[:-1])
            quantile_factor = df["factor_quantile"]
            quantile_turnover = pd.concat([perf.quantile_turnover(quantile_factor, q, period)
                                        for q in quantile_factor.sort_values().unique().tolist()],axis=1,).iloc[::period, :]
            quantile_turnover['year'] = quantile_turnover.index.year
            ltvr = quantile_turnover[[5, 'year']].groupby(['year'])[5].mean()
            stvr = quantile_turnover[[1, 'year']].groupby(['year'])[1].mean()
            lstvr = ltvr + stvr
            return pd.DataFrame({'ltvr': ltvr, 'stvr': stvr, 'lstvr': lstvr})

        df_tvr = turnover(self.combine_data, period)

        # IC IR 计算公式待定
        def ic(df, period):
            df['year'] = df.index.get_level_values('date').year
            allic = df.groupby('year').apply(lambda group: spearmanr(group['factor'].iloc[::eval(period[:-1])], group[period].iloc[::eval(period[:-1])]).correlation)
            lic = df[df.factor_quantile == 5].groupby('year').apply(lambda group: spearmanr(group['factor'].iloc[::eval(period[:-1])], group[period].iloc[::eval(period[:-1])]).correlation)
            sic = df[df.factor_quantile == 1].groupby('year').apply(lambda group: spearmanr(group['factor'].iloc[::eval(period[:-1])], group[period].iloc[::eval(period[:-1])]).correlation)
            return pd.DataFrame({'allic': allic, 'lic': lic, 'sic': sic})

        df_ic = ic(self.combine_data, period)

        return pd.concat([df_ret, df_tvr, df_ic], axis=1)
    
    def factor_statistic(self):
        # 描述性统计
        descriptive_stats_total = self.combine_data['factor'].describe()
        stats_df = self.combine_data.groupby(self.combine_data.index.get_level_values('date').year).apply(lambda x: x['factor'].describe())
        # 缺失值
        null_total = self.combine_data['factor'].isnull().sum()
        null_df = pd.DataFrame(self.combine_data.groupby(self.combine_data.index.get_level_values('date').year).apply(lambda x: x['factor'].isnull().sum()), columns=['null'])
        # 异常值
        def outliers_stats(df):
            df['year'] = df.index.get_level_values('date').year
            yearly_stats = {}

            # 遍历每个年份，计算样本数和异常值数量
            for year, group in df.groupby('year'):
                Q1 = group['factor'].quantile(0.25)
                Q3 = group['factor'].quantile(0.75)
                IQR = Q3 - Q1
                outliers = group[(group['factor'] < (Q1 - 1.5 * IQR)) | (group['factor'] > (Q3 + 1.5 * IQR))]
                
                # 添加结果到字典
                yearly_stats[year] = {
                    'Outliers': outliers.shape[0],
                    'Total Samples': group.shape[0]
                }
            return pd.DataFrame(yearly_stats).T

        outlier_df = outliers_stats(self.combine_data)
        return pd.concat([stats_df, null_df, outlier_df], axis=1)

    def return_analysis(self):
        tears.create_returns_tear_sheet(self.combine_data, self.long_short)
    
    def ic_analysis(self):
        tears.create_information_tear_sheet(self.combine_data, self.long_short)

    def turnover_analysis(self):
        tears.create_turnover_tear_sheet(self.combine_data)
    
    def statistic_analysis(self):
        plot.plot_quantile_statistics_table(self.combine_data)
        
    def analysis(self) -> None:
        if self.instructors == 'all' or 'all' in self.instructors:
            tears.create_full_tear_sheet(self.factor_data, long_short=self.long_short)
        
        # plot.plot_quantile_statistics_table(self.factor_data)
        # tears.create_returns_tear_sheet(self.factor_data, self.long_short)
        # tears.create_information_tear_sheet(self.factor_data, self.long_short)
        # tears.create_turnover_tear_sheet(self.factor_data)
        
