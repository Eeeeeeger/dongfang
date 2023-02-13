from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from TradingUtils.strategy import *
from TradingUtils.dataflow import *


class baseContract(ABC):
    @abstractmethod
    def concat(self):
        pass

    @abstractmethod
    def trade(self):
        pass


class DirectContract(baseContract):
    def __init__(self, data_path: str or Path = './input/T/'):
        self.data_path = Path(data_path).absolute()
        self.data_dict = {}
        for x in self.data_path.glob("*.csv"):
            temp = pd.read_csv(x, index_col=0)
            temp.index.name = 'date'
            temp.index = pd.to_datetime(temp.index)
            temp = (temp
                    .rename(columns=lambda x: x.lower())
                    .assign(code=x.stem,
                            ret=lambda x: x['settle'] / x['pre_settle'] - 1
                            )
                    .set_index('code', append=True)
                    )
            self.data_dict[x.stem] = temp

        self.contract_list = list(self.data_dict.keys())
        self.full_data = pd.concat(self.data_dict.values(), axis=0)
        self.tradingdays_list = self.full_data.index.get_level_values('date').unique().to_list()
        self.tradingdays_list.sort()
        logger.info(f"init {self.__class__.__name__} ")

    def concat(self, indicator: str = 'oi', commission: float = 0.0001) -> pd.DataFrame:
        logger.info(f"concat {self.__class__.__name__} with indicator: {indicator} and commission: {commission} ")
        refer = self.full_data[indicator].unstack()
        self.contract_each_day = refer.idxmax(axis=1).shift(1).bfill().rename('code')
        self.contract_change_day = self.contract_each_day.reset_index().groupby('code').first()['date']
        self.dominant_list = self.contract_change_day.index.tolist()
        self.dominant_data = pd.merge(self.contract_each_day.to_frame().set_index('code', append=True),
                                      self.full_data[['ret']],
                                      left_index=True,
                                      right_index=True,
                                      how='left').fillna(0)
        # 开平仓万一
        self.dominant_data.loc[self.contract_change_day.to_list()[1:]] -= commission
        self.dominant_data['cum_ret'] = (self.dominant_data['ret'] + 1).cumprod()
        self.trade_info = self.contract_change_day.to_frame('trade_date').reset_index().assign(weight=1)
        self.trade_info.columns = ['sec_code', 'trade_date', 'weight']
        self.trade_info = self.trade_info[self.trade_info['weight'] != 0.0]
        self.trade_info['trade_date'] = self.trade_info['trade_date'].apply(lambda x: self.tradingdays_list[self.tradingdays_list.index(x)-1])
        self.trade_info.loc[0, 'trade_date'] = self.tradingdays_list[0]
        self.trade_info.to_csv(f'./input/signal/trade_{self.__class__.__name__}.csv')

    def trade(self):
        cerebro = bt.Cerebro()
        cols = ['pre_close', 'open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'pre_settle', 'settle',
                'openinterest']
        lt = []
        logger.info('start FuturesPandasData read... ')
        for x in Path('./input/quote/').glob('*.csv'):
            df = pd.read_csv(x, index_col=0)
            df.index = pd.to_datetime(df.index)
            df.columns = cols
            df = df.ffill()
            df['sec_code'] = x.stem
            # 结算价计算每日收益
            df['close'] = df['settle']
            lt.append(df)
        daily_info = pd.concat(lt, axis=0)
        date_list = daily_info.index.unique().sort_values()

        for code in daily_info['sec_code'].unique():
            temp_df = pd.DataFrame(index=date_list)
            df = daily_info.query(f" sec_code == '{code}' ")[cols]
            df = pd.merge(temp_df, df, left_index=True, right_index=True, how='left').fillna(0)
            cerebro.adddata(FuturesPandasData(dataname=df), name=code)

        cerebro.addstrategy(FuturesRollingStrategy, **{'reserve': 0.02,
                                                       'signal_df': self.trade_info})

        cerebro.broker.setcash(100000000.0)
        cerebro.broker.setcommission(
            commission=3,
            # margin=0.02,
            automargin=0.02 * 10000,
            mult=10000,
            commtype=bt.comminfo.CommInfoBase.COMM_FIXED,
            stocklike=False,

        )
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
        logger.info('start to run strategy...')
        result = cerebro.run()

        pnl = pd.Series(result[0].analyzers._TimeReturn.get_analysis()).dropna().to_frame('ret')
        pnl.index.name = 'date'
        self.trade_data = pnl
        self.trade_data['cum_ret'] = (self.trade_data['ret'].fillna(0) + 1).cumprod()


class AfterContract(DirectContract):
    def __init__(self, data_path: str or Path = './input/T/'):
        super().__init__(data_path)

    def concat(self, indicator: str = 'oi', commission: float = 0.0001, days: int = 3) -> pd.DataFrame:
        logger.info(
            f"concat {self.__class__.__name__} with indicator: {indicator}, commission: {commission} and days: {days} ")
        refer = self.full_data[indicator].unstack()
        self.contract_each_day = refer.idxmax(axis=1).shift(days + 1).bfill().rename('code')
        self.contract_change_day = self.contract_each_day.reset_index().groupby('code').first()['date']
        self.dominant_list = self.contract_change_day.index.tolist()
        self.dominant_data = pd.merge(self.contract_each_day.to_frame().set_index('code', append=True),
                                      self.full_data[['ret']],
                                      left_index=True,
                                      right_index=True,
                                      how='left').fillna(0)
        # 开平仓万一
        self.dominant_data.loc[self.contract_change_day.to_list()[1:]] -= commission
        self.dominant_data['cum_ret'] = (self.dominant_data['ret'] + 1).cumprod()
        self.trade_info = self.contract_change_day.to_frame('trade_date').reset_index().assign(weight=1)
        self.trade_info.columns = ['sec_code', 'trade_date', 'weight']
        self.trade_info = self.trade_info[self.trade_info['weight'] != 0.0]
        self.trade_info['trade_date'] = self.trade_info['trade_date'].apply(lambda x: self.tradingdays_list[self.tradingdays_list.index(x)-1])
        self.trade_info.loc[0, 'trade_date'] = self.tradingdays_list[0]
        self.trade_info.to_csv(f'./input/signal/trade_{self.__class__.__name__}.csv')
        return self.dominant_data


class BeforeContract(DirectContract):
    def __init__(self, data_path: str or Path = './input/T/'):
        super().__init__(data_path)

    def concat(self, indicator: str = 'oi', commission: float = 0.0001, days: int = 3) -> pd.DataFrame:
        logger.info(
            f"concat {self.__class__.__name__} with indicator: {indicator}, commission: {commission} and days: {days} ")

        refer = self.full_data[indicator].unstack()
        self.contract_each_day = refer.idxmax(axis=1).shift(1 - days).bfill().rename('code')
        self.contract_change_day = self.contract_each_day.reset_index().groupby('code').first()['date']
        self.dominant_list = self.contract_change_day.index.tolist()
        self.dominant_data = pd.merge(self.contract_each_day.to_frame().set_index('code', append=True),
                                      self.full_data[['ret']],
                                      left_index=True,
                                      right_index=True,
                                      how='left').fillna(0)
        # 开平仓万一
        self.dominant_data.loc[self.contract_change_day.to_list()[1:]] -= commission
        self.dominant_data['cum_ret'] = (self.dominant_data['ret'] + 1).cumprod()
        self.trade_info = self.contract_change_day.to_frame('trade_date').reset_index().assign(weight=1)
        self.trade_info.columns = ['sec_code', 'trade_date', 'weight']
        self.trade_info = self.trade_info[self.trade_info['weight'] != 0.0]
        self.trade_info['trade_date'] = self.trade_info['trade_date'].apply(lambda x: self.tradingdays_list[self.tradingdays_list.index(x)-1])
        self.trade_info.loc[0, 'trade_date'] = self.tradingdays_list[0]
        self.trade_info.to_csv(f'./input/signal/trade_{self.__class__.__name__}.csv')
        return self.dominant_data
