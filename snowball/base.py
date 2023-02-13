from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from utils import *
from WindPy import *
from pathlib import Path
from constants import INDEX_CODE
from collections import OrderedDict
from loguru import logger

class baseStructure(object):

    def __init__(self, name: str, underlying: str, start_date: str or pd.Timestamp, maturity: int):
        self.name = name
        self.underlying = underlying
        if self.underlying not in INDEX_CODE.keys():
            raise ValueError("invalid underlying")
        self.code = INDEX_CODE[underlying]

        self.start_date = pd.Timestamp(start_date)
        if not isTradingDay(self.start_date):
            raise ValueError("start date is not a trading day")
        self.maturity = maturity
        self.end_date = self.start_date + relativedelta(months=self.maturity)
        if not isTradingDay(self.end_date):
            self.end_date = prevTradingDay(self.end_date)
        self.trading_days = tradingDays(self.start_date, self.end_date)

        self.cur_date = self.start_date
        self.price_series = pd.read_csv(f'./input/{underlying}.csv', index_col=0)
        self.price_series.index = pd.to_datetime(self.price_series.index)
        self.start_price = self.price_series.loc[self.cur_date, 'close']
        self.cur_price = self.start_price


class snowBall(baseStructure):

    def __init__(self, name, underlying, start_date, maturity, upper, lower, ret):
        super().__init__(name, underlying, start_date, maturity)
        self.upper = upper
        self.lower = lower
        self.coupon = ret
        if self.underlying in ['10Y']:
            self.upper_price = self.start_price + self.upper
            self.lower_price = self.start_price + self.lower
        else:
            self.upper_price = self.start_price * self.upper
            self.lower_price = self.start_price * self.lower
        self.expire_date = None
        self.expire_flag, self.knockin_flag, self.knockout_flag = False, False, False
        self.knockin_date, self.knockin_price = None, None
        self.knockout_date, self.knockout_price = None, None
        self.valid_knockout_date = [
            temp_date if isTradingDay(temp_date := self.start_date + relativedelta(months=ii)) else prevTradingDay(
                temp_date) for ii in range(1, self.maturity+1)]
        self.duration_day = 0
        self.duration_month = 0
        logger.info(f'{self.name} initializing successfully')

    def check_knockin(self):
        if self.knockin_flag:
            return
        if self.cur_price < self.lower_price:
            self.knockin_flag = True
            self.knockin_date = self.cur_date
            self.knockin_price = self.cur_price
            logger.info(f'{self.name} knock in at {self.knockin_price} in {self.knockin_date}')
        return

    def check_knockout(self):
        if self.knockout_flag:
            return
        if self.cur_date in self.valid_knockout_date:
            self.duration_month += 1
            if self.cur_price > self.upper_price:
                self.knockout_flag = True
                self.knockout_date = self.cur_date
                self.knockout_price = self.cur_price
                self.expire_date = self.cur_date
                self.expire_flag = True
                self.ret = self.coupon * self.duration_month / 12
                self.annual_ret = self.ret / self.duration_month * 12
                logger.info(f'{self.name} knock out at {self.knockout_price} in {self.knockout_date}')

        return

    def update_daily(self):
        for cur_date in self.trading_days:
            self.duration_day += 1
            if self.expire_flag:
                logger.info(f'{self.name} expire at {self.expire_date} in advance')
                break
            self.cur_date = cur_date
            self.cur_price = self.price_series.loc[self.cur_date, 'close']
            self.check_knockin()
            self.check_knockout()
        if not self.expire_flag:
            self.expire_date = self.end_date
            self.expire_flag = True
            logger.info(f'{self.name} expire at {self.expire_date} normally')
            if self.knockin_flag:
                self.ret = 0 if (self.cur_price >= self.start_price) else (self.cur_price - self.start_price) / self.start_price
                self.annual_ret = self.ret / self.duration_month * 12
            else:
                self.ret = self.coupon * self.duration_month / 12
                self.annual_ret = self.ret * 12 / self.duration_month

    def save_results(self,
                     output_path: str or Path = Path('./output/snowball.csv')
                     ):
        temp_df = pd.DataFrame({'产品名称': self.name,
                                '挂钩标的': self.underlying,
                                '标的代码': self.code,
                                '结构': self.__class__.__name__,
                                '成立日': self.start_date,
                                '初始点位': self.start_price,
                                '到期日': self.end_date,
                                '期限': '{}个月'.format(self.maturity),
                                '敲出阈值': self.upper,
                                '敲出价格': self.upper_price,
                                '敲入阈值': self.lower,
                                '敲入价格': self.lower_price,
                                '票息': self.coupon,
                                '状态': '已敲出未曾敲入' if self.knockout_flag and not self.knockin_flag else '已敲出曾敲入' if self.knockout_flag and self.knockin_flag else '已敲入且到期' if self.knockin_flag else '未敲入且到期',
                                '敲入日期': self.knockin_date,
                                '敲入点位': self.knockin_price,
                                '敲出日期': self.knockout_date,
                                '敲出点位': self.knockout_price,
                                '存续时长(日)': self.duration_day,
                                '存续时长(月)': self.duration_month,
                                '存续时长(年)': self.duration_month / 12,
                                '实际损益': self.ret,
                                '实际年化损益': self.annual_ret,
                                '标的资产实际损益': self.price_series.loc[self.expire_date, 'close']/self.start_price - 1,
                                '标的资产年化损益': (self.price_series.loc[self.expire_date, 'close']/self.start_price - 1)*12/self.duration_month,
                                },
                               index=[0]).set_index('产品名称')
        output_path = Path(output_path).absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            temp_df.to_csv(output_path, mode='a', header=None, encoding='utf-8-sig')
        else:
            temp_df.to_csv(output_path, encoding='utf-8-sig')


if __name__ == "__main__":

    for ii,d in enumerate(tradingDays('20170101', '20220131')):
        a = snowBall(f'产品{ii}', 'ZZ500', d, 12, 1.0, 0.8, 0.2)
        a.update_daily()
        a.save_results('./output/ZZ500.csv')
