import pandas as pd
import numpy as np
from .BaseContract import baseContract, DirectContract
from pathlib import Path
from loguru import logger


class AfterMovingContract(DirectContract):
    def __init__(self, data_path: str or Path = './input/T/'):
        super().__init__(data_path)

    def concat(self, indicator: str = 'oi', commission: float = 0.0001, days: int = 3) -> pd.DataFrame:
        logger.info(
            f"concat {self.__class__.__name__} with indicator: {indicator}, commission: {commission} and days: {days} ")
        refer = self.full_data[indicator].unstack()
        self.contract_each_day = refer.idxmax(axis=1).shift(1).bfill().rename('code')
        self.contract_change_day = self.contract_each_day.reset_index().groupby('code').first()['date']
        self.dominant_list = self.contract_change_day.index.tolist()
        self.dominant_data = pd.merge(self.contract_each_day.to_frame().set_index('code', append=True),
                                      self.full_data[['ret']],
                                      left_index=True,
                                      right_index=True,
                                      how='left').fillna(0)

        self.prev_dominant_dict = dict(zip(self.dominant_list[1:], self.dominant_list[:-1]))
        self.prev_dominant_dict[self.dominant_list[0]] = self.dominant_list[0]
        self.trade_info = pd.DataFrame(index=[0],
                                       columns=['sec_code', 'trade_date', 'weight'],
                                       data=[[self.dominant_list[0], self.contract_change_day.to_list()[0], 1]])
        # 最开始不用逐步移仓
        for day in self.contract_change_day.to_list()[1:]:
            indice = self.tradingdays_list.index(day)
            for ii, each_day in enumerate(self.tradingdays_list[indice:indice + days]):
                main_contract = self.contract_each_day[each_day]
                prev_contract = self.prev_dominant_dict[main_contract]
                '''
                移仓逻辑：
                1. OI最大的第二天主力合约切换；
                2. 主力切换当天 按1/days比例切换到下一个主力合约, 1-1/days仍然是原来的主力；
                3. 第days天完全切换。
                '''
                self.dominant_data.loc[(each_day, main_contract), 'ret'] = (
                        (1 - (ii + 1) / days) * self.full_data.loc[(each_day, prev_contract), 'ret']
                        + (ii + 1) / days * self.dominant_data.loc[(each_day, main_contract), 'ret']
                        - commission / days
                )
                self.trade_info = self.trade_info.append(
                    pd.Series({'sec_code': prev_contract, 'trade_date': each_day, 'weight': (1 - (ii + 1) / days)}),
                    ignore_index=True)
                self.trade_info = self.trade_info.append(
                    pd.Series({'sec_code': main_contract, 'trade_date': each_day, 'weight': (ii + 1) / days}),
                    ignore_index=True)

        self.dominant_data['cum_ret'] = (self.dominant_data['ret'] + 1).cumprod()
        self.trade_info = self.trade_info[self.trade_info['weight'] != 0.0]
        self.trade_info['trade_date'] = self.trade_info['trade_date'].apply(lambda x: self.tradingdays_list[self.tradingdays_list.index(x)-1])
        self.trade_info.loc[0, 'trade_date'] = self.tradingdays_list[0]
        self.trade_info.to_csv(f'./input/signal/trade_{self.__class__.__name__}.csv')
        return self.dominant_data


class BeforeMovingContract(DirectContract):
    def __init__(self, data_path: str or Path = './input/T/'):
        super().__init__(data_path)

    def concat(self, indicator: str = 'oi', commission: float = 0.0001, days: int = 3) -> pd.DataFrame:
        logger.info(
            f"concat {self.__class__.__name__} with indicator: {indicator}, commission: {commission} and days: {days} ")
        refer = self.full_data[indicator].unstack()
        self.contract_each_day = refer.idxmax(axis=1).shift(1).bfill().rename('code')
        self.contract_change_day = self.contract_each_day.reset_index().groupby('code').first()['date']
        self.dominant_list = self.contract_change_day.index.tolist()
        self.dominant_data = pd.merge(self.contract_each_day.to_frame().set_index('code', append=True),
                                      self.full_data[['ret']],
                                      left_index=True,
                                      right_index=True,
                                      how='left').fillna(0)

        self.next_dominant_dict = dict(zip(self.dominant_list[:-1], self.dominant_list[1:]))
        self.next_dominant_dict[self.dominant_list[-1]] = self.dominant_list[-1]
        self.trade_info = pd.DataFrame(index=[0],
                                       columns=['sec_code', 'trade_date', 'weight'],
                                       data=[[self.dominant_list[0], self.contract_change_day.to_list()[0], 1]])

        # 最开始不用逐步移仓
        for day in self.contract_change_day.to_list()[1:]:
            indice = self.tradingdays_list.index(day)
            for ii, each_day in enumerate(self.tradingdays_list[indice - days:indice]):
                main_contract = self.contract_each_day[each_day]
                next_contract = self.next_dominant_dict[main_contract]
                '''
                移仓逻辑：
                1. OI最大的第二天主力合约切换；
                2. 主力切换前days天, 按1/days比例切换到下一个主力合约, 1-1/days仍然是原来当前主力；
                3. OI最大的第二天完全切换。
                '''
                self.dominant_data.loc[(each_day, main_contract), 'ret'] = (
                        (ii + 1) / days * self.full_data.loc[(each_day, next_contract), 'ret']
                        + (1 - (ii + 1) / days) * self.dominant_data.loc[(each_day, main_contract), 'ret']
                        - commission / days
                )
                self.trade_info = self.trade_info.append(
                    pd.Series({'sec_code': main_contract, 'trade_date': each_day, 'weight': (1 - (ii + 1) / days)}),
                    ignore_index=True)
                self.trade_info = self.trade_info.append(
                    pd.Series({'sec_code': next_contract, 'trade_date': each_day, 'weight': (ii + 1) / days}),
                    ignore_index=True)

        self.dominant_data['cum_ret'] = (self.dominant_data['ret'] + 1).cumprod()
        self.trade_info = self.trade_info[self.trade_info['weight'] != 0.0]
        self.trade_info['trade_date'] = self.trade_info['trade_date'].apply(lambda x: self.tradingdays_list[self.tradingdays_list.index(x)-1])
        self.trade_info.loc[0, 'trade_date'] = self.tradingdays_list[0]
        self.trade_info.to_csv(f'./input/signal/trade_{self.__class__.__name__}.csv')
        return self.dominant_data
