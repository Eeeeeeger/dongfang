import pandas as pd

from .BaseModel import *
from statsmodels.api import OLS, WLS
import statsmodels.api as sm
from loguru import logger


class OLSModel(BaseModel):
    def __init__(self,
                 dominant_type: str = 'AfterMovingContract',
                 trade_type: bool = False,
                 days: int = 3,
                 ):
        super(OLSModel, self).__init__(dominant_type, trade_type, days)

    @staticmethod
    def ols(X, y, const=True):
        if const:
            X = sm.add_constant(X)
        lr = OLS(endog=y, exog=X).fit()
        return lr.params[-1], lr.rsquared

    def calc_hedge_ratio(self, trace_back: int = 30, *args, **kwargs):
        self.reg_result = pd.DataFrame(index=self.reg_data.index, columns=['ratio', 'adj'])
        for ii in range(trace_back, len(self.reg_data)):
            # auto-shift
            self.reg_result['ratio'].iloc[ii] = self.ols(
                self.reg_data.iloc[ii - trace_back:ii]['futures'],
                self.reg_data.iloc[ii - trace_back:ii]['bond'],
                const=kwargs.get('const', True))[0]

            # vol_adj = 1.2 if self.reg_data['bond'].iloc[ii - trace_back:ii].std()*np.sqrt(252) >= 0.2 else 1
            # self.reg_result['ratio'].iloc[ii] = self.reg_result['ratio'].iloc[ii] * vol_adj
            adj_days = kwargs.get('adj_days', 10)
            if kwargs.get('adj_method', 'trend') == 'trend':
                self.reg_result['adj'].iloc[ii] = kwargs.get('leverage', 1.2) if self.ols(
                    np.array(range(adj_days)),
                    self.reg_data.iloc[ii - adj_days:ii]['bond'],
                    const=kwargs.get('const', True))[1] > kwargs.get('threshold', 1) else 1
            elif kwargs.get('adj_method', 'trend') == 'std':
                self.reg_result['adj'].iloc[ii] = kwargs.get('leverage', 1.2) if self.reg_data['bond'].iloc[
                                                                                 ii - adj_days:ii].std()*np.sqrt(252) > kwargs.get(
                    'threshold', 1) else 1
            else:
                self.reg_result['adj'].iloc[ii] = 1

            self.reg_result['ratio'].iloc[ii] = self.reg_result['ratio'].iloc[ii] * self.reg_result['adj'].iloc[ii]
        self.reg_result.dropna(inplace=True)

    def ajust_hedge_frequency(self, frequency: str or int = 'daily', commission: float = 0.0001, *args, **kwargs):
        self.check_frequency(frequency=frequency)
        if isinstance(frequency, str):
            if frequency == 'daily':
                self.hedge_result = self.reg_result
            elif frequency == 'weekly':
                pass
            elif frequency == 'monthly':
                self.hedge_result = (pd.DataFrame(index=self.reg_result.index)
                                     .merge(
                    (self.reg_result
                    .reset_index()
                    .groupby([self.reg_result.index.year, self.reg_result.index.month],
                             as_index=False)
                    .first()
                    .set_index('date')['ratio']),
                    left_index=True,
                    right_index=True,
                    how='left',
                ).ffill())
            else:
                self.hedge_result = (pd.DataFrame(index=self.reg_result.index)
                                     .merge(
                    (self.reg_result
                    .reset_index()
                    .groupby(self.reg_result.index.year,
                             as_index=False)
                    .first()
                    .set_index('date')['ratio']),
                    left_index=True,
                    right_index=True,
                    how='left',
                ).ffill())
        if isinstance(frequency, int):
            self.hedge_result = (pd.DataFrame(index=self.reg_result.index)
                                 .merge(self.reg_result.iloc[::frequency],
                                        left_index=True,
                                        right_index=True,
                                        how='left',
                                        )
                                 ).ffill()
        # 对冲后的收益计算
        logger.info(f'calc return after hedging with {commission} ')
        self.hedge_result = self.reg_data.merge(self.hedge_result, left_index=True, right_index=True, how='right')
        self.hedge_result['hedge'] = (
                self.hedge_result['bond'] - self.hedge_result['ratio'] * self.hedge_result['futures']
            # - self.hedge_result['ratio'].fillna(0).diff().abs().fillna(0) * commission
        ).astype(np.float64)

    def hedge(self, frequency: str or int = 'daily', trace_back: int = 30, commission: float = 0.0001, *args, **kwargs):
        '''
        生成对冲曲线
        Args:
            frequency: 对冲频率
            trace_back:  回归回溯天数
            commission: 手续费
        '''

        self.reg_data = self.bond[['ret']].merge(self.futures['ret'], left_index=True, right_index=True, how='inner')
        self.reg_data.columns = ['bond', 'futures']
        # 计算对冲比率
        logger.info(f'start calc hedge ratio by {self.__class__.__name__} ')
        self.calc_hedge_ratio(trace_back=trace_back, *args, **kwargs)
        # 根据对冲频率调整对冲比率
        logger.info(
            f'deal with hedge frequency {frequency} with start_date: {self.reg_result.index[0].date()}, end_date: {self.reg_result.index[-1].date()}')
        self.ajust_hedge_frequency(frequency=frequency, commission=commission, *args, **kwargs)
        return self.summary_info(self.hedge_result,
                                 **{'save': True,
                                    'name': f'{self.__class__.__name__}_{frequency}_{trace_back}_{self.trade_type}'})


class WLSModel(OLSModel):
    def __init__(self,
                 dominant_type: str = 'AfterMovingContract',
                 trade_type: bool = False,
                 days: int = 3,
                 ):
        super(WLSModel, self).__init__(dominant_type, trade_type, days)

    @staticmethod
    def wls(X, y, halflife, const=True):
        if const:
            X = sm.add_constant(X)
        lr = WLS(endog=y, exog=X,
                 weights=pd.Series([1] + [0 for _ in range(len(X) - 1)]).ewm(halflife=halflife,
                                                                             adjust=False).mean().values[
                         ::-1]).fit()
        return lr.params[-1], lr.rsquared

    def calc_hedge_ratio(self, trace_back: int = 30, halflife: int = 20, *args, **kwargs):
        self.reg_result = pd.DataFrame(index=self.reg_data.index, columns=['ratio', 'adj'])
        for ii in range(trace_back, len(self.reg_data)):
            # auto-shift
            self.reg_result['ratio'].iloc[ii] = self.wls(self.reg_data.iloc[ii - trace_back:ii]['futures'],
                                                         self.reg_data.iloc[ii - trace_back:ii]['bond'],
                                                         halflife=halflife,
                                                         const=kwargs.get('const', True))[0]
            adj_days = kwargs.get('adj_days', 10)
            if kwargs.get('adj_method', 'trend') == 'trend':
                self.reg_result['adj'].iloc[ii] = kwargs.get('leverage', 1.2) if self.ols(
                    np.array(range(adj_days)),
                    self.reg_data.iloc[ii - adj_days:ii]['bond'],
                    const=kwargs.get('const', True))[1] > kwargs.get('threshold', 1) else 1
            elif kwargs.get('adj_method', 'trend') == 'std':
                self.reg_result['adj'].iloc[ii] = kwargs.get('leverage', 1.2) if self.reg_data['bond'].iloc[
                                                                                 ii - adj_days:ii].std()*np.sqrt(252) > kwargs.get(
                    'threshold', 1) else 1
            else:
                self.reg_result['adj'].iloc[ii] = 1
            self.reg_result['ratio'].iloc[ii] = self.reg_result['ratio'].iloc[ii] * self.reg_result['adj'].iloc[ii]
        self.reg_result.dropna(inplace=True)

    def hedge(self, frequency: str or int = 'daily', trace_back: int = 30, commission: float = 0.0001,
              halflife: int = 20, *args, **kwargs):
        '''
        生成对冲曲线
        Args:
            frequency: 对冲频率
            trace_back:  回归回溯天数
            commission: 手续费
        '''

        self.reg_data = self.bond[['ret']].merge(self.futures['ret'], left_index=True, right_index=True, how='inner')
        self.reg_data.columns = ['bond', 'futures']
        # 计算对冲比率
        logger.info(f'start calc hedge ratio by {self.__class__.__name__} ')
        self.calc_hedge_ratio(trace_back=trace_back, halflife=halflife, *args, **kwargs)
        # 根据对冲频率调整对冲比率
        logger.info(
            f'deal with hedge frequency {frequency} with start_date: {self.reg_result.index[0].date()}, end_date: {self.reg_result.index[-1].date()}')
        self.ajust_hedge_frequency(frequency=frequency, commission=commission, *args, **kwargs)
        return self.summary_info(self.hedge_result,
                                 **{'save': True, 'name': f'{self.__class__.__name__}_{frequency}_{trace_back}'})


class VARModel(OLSModel):
    def __init__(self,
                 dominant_type: str = 'AfterMovingContract',
                 trade_type: bool = False,
                 days: int = 3,
                 ):
        super(VARModel, self).__init__(dominant_type, trade_type, days)

    # @staticmethod
    # def var(X, y, const=True):
    #     X = pd.concat([X.shift(1), X.shift(2), y.shift(1), y.shift(2), X], axis=1)
    #     if const:
    #         X = sm.add_constant(X)
    #     lt = []
    #     for reg_X in [X.iloc[:, [0, 2, -1]], X.iloc[:, [0, 1, 2, -1]], X.iloc[:, [0, 2, 3, -1]],
    #                   X.iloc[:, [0, 1, 2, 3, -1]]]:
    #         lr = OLS(endog=y, exog=reg_X, missing='drop').fit()
    #         lt.append([lr.params[-1], lr.rsquared, lr.aic, lr.bic])
    #     res = pd.DataFrame(lt)
    #     return res.loc[res[1].idxmin()][0]

    @staticmethod
    def var(X, y, const=True):
        X = pd.concat([X.shift(1), X.shift(2), y.shift(1), y.shift(2), X], axis=1)
        if const:
            X = sm.add_constant(X)
        lr = OLS(endog=y, exog=X, missing='drop').fit()
        return lr.params[-1], lr.rsquared

    def calc_hedge_ratio(self, trace_back: int = 30, *args, **kwargs):
        self.reg_result = pd.DataFrame(index=self.reg_data.index, columns=['ratio', 'adj'])
        for ii in range(trace_back, len(self.reg_data)):
            # auto-shift
            self.reg_result['ratio'].iloc[ii] = self.var(self.reg_data.iloc[ii - trace_back:ii]['futures'],
                                                         self.reg_data.iloc[ii - trace_back:ii]['bond'],
                                                         const=kwargs.get('const', True))[0]
            adj_days = kwargs.get('adj_days', 10)
            if kwargs.get('adj_method', 'trend') == 'trend':
                self.reg_result['adj'].iloc[ii] = kwargs.get('leverage', 1.2) if self.ols(
                    np.array(range(adj_days)),
                    self.reg_data.iloc[ii - adj_days:ii]['bond'],
                    const=kwargs.get('const', True))[1] > kwargs.get('threshold', 1) else 1
            elif kwargs.get('adj_method', 'trend') == 'std':
                self.reg_result['adj'].iloc[ii] = kwargs.get('leverage', 1.2) if self.reg_data['bond'].iloc[
                                                                                 ii - adj_days:ii].std()*np.sqrt(252) > kwargs.get(
                    'threshold', 1) else 1
            else:
                self.reg_result['adj'].iloc[ii] = 1
            self.reg_result['ratio'].iloc[ii] = self.reg_result['ratio'].iloc[ii] * self.reg_result['adj'].iloc[ii]
        self.reg_result.dropna(inplace=True)
