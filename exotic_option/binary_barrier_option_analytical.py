# encoding: utf-8

import numpy as np
import pandas as pd
from scipy import stats


class BinaryBarrierOptionAnalytical():
    """
    binary barrier option
    参考《The Complete Guide to Option》 p176
    """

    def __init__(self, option_type, barrier_up_or_down, barrier_in_or_out, asset_or_cash,
                 spot_px, strike_px, barrier_px, rebate_px, years_rf,
                 rf_rate, volatility, dvd_rate, barrier_state='Unknocked', years_vola=None,
                 **kwargs):
        """
        初始化
        :param option_type: 期权类型 ('Call', 'Put')
        :param barrier_up_or_down: ('Up', 'Down')
        :param barrier_in_or_out: ('In', 'Out')
        :param asset_or_cash: ('Asset', 'Cash')
        :param spot_px: 即期价格
        :param strike_px: 行权价格
        :param barrier_px: 障碍价格
        :param rebate_px: （结束价格超过行权价格后）收益金额, rebate
        :param years_rf: 折现年数
        :param rf_rate: 无风险利率（连续复利）
        :param volatility:  年化波动率
        :param dvd_rate: 分红利率（连续复利）
        :param barrier_state: 障碍状态
        :param years_vola: 波动年数
        """
        if option_type not in ('Call', 'Put', 'At-hit', 'At-expiration'):
            raise ValueError
        if barrier_up_or_down not in ('Up', 'Down'):
            raise ValueError
        if barrier_in_or_out not in ('In', 'Out'):
            raise ValueError
        if asset_or_cash not in ('Asset', 'Cash'):
            raise ValueError

        self.option_type = option_type
        self.barrier_up_or_down = barrier_up_or_down
        self.barrier_in_or_out = barrier_in_or_out
        self.asset_or_cash = asset_or_cash
        self.spot_px = spot_px
        self.strike_px = strike_px
        self.barrier_px = barrier_px
        self.rebate_px = rebate_px
        self.years_rf = years_rf
        self.years_vola = years_vola
        self.rf_rate = rf_rate
        self.volatility = volatility
        self.dvd_rate = dvd_rate
        self.barrier_state = barrier_state
        # 由于定义法计算theta时会改变rf的值，cost_of_carry也应随时改变，所以每次使用cost_of_carry时都应当调用该函数
        if self.barrier_in_or_out == 'In' and self.option_type == 'At-hit' and self.asset_or_cash == 'Asset':
            self.rebate_px = self.barrier_px
        super(BinaryBarrierOptionAnalytical, self).__init__(**kwargs)

    def get_px(self):
        spot_px = self.spot_px
        strike_px = self.strike_px
        barrier_px = self.barrier_px
        years_rf = self.years_rf
        years_vola = self.years_vola if self.years_vola is not None else self.years_rf
        option_type = self.option_type
        barrier_up_or_down = self.barrier_up_or_down
        barrier_in_or_out = self.barrier_in_or_out
        asset_or_cash = self.asset_or_cash
        rf_rate = self.rf_rate
        dvd_rate = self.dvd_rate
        volatility = self.volatility
        rebate_px = self.rebate_px
        barrier_state = self.barrier_state

        # if self.event == BarrierEventEnum.KnockOut.value and self.barrierType in (
        # BarrierTypeEnum.UpOut, BarrierTypeEnum.DownOut):
        #     return self.rebateAmount
        # elif self.event == BarrierEventEnum.KnockIn.value and self.barrierType in (
        # BarrierTypeEnum.UpIn, BarrierTypeEnum.DownIn):
        #     return VanillaEuropeanOption(self.optionType, ExerciseTypeEnum.European,
        #                                  spot_px, strike_px, self.maturity_years, rf_rate, dvd_rate,
        #                                  self.volatility).get_analytical_px()

        a1, b1, a2, b2, a3, b3, a4, b4, a5 = self._get_factors(spot_px=spot_px, strike_px=strike_px,
                                                               barrier_px=barrier_px,
                                                               years_rf=years_rf, years_vola=years_vola,
                                                               option_type=option_type,
                                                               barrier_up_or_down=barrier_up_or_down,
                                                               barrier_in_or_out=barrier_in_or_out,
                                                               rf_rate=rf_rate, dvd_rate=dvd_rate,
                                                               volatility=volatility, rebate_px=rebate_px)

        if barrier_in_or_out == 'In' and option_type == 'At-hit': # 1 2 3 4
            px = a5
        elif barrier_in_or_out == 'In' and option_type == 'At-expiration' and asset_or_cash == 'Cash': # 5 6
            px = b2 + b4
        elif barrier_in_or_out == 'In' and option_type == 'At-expiration' and asset_or_cash == 'Asset': # 7 8
            px = a2 + a4
        elif barrier_in_or_out == 'Out' and option_type in ('At-hit', 'At-expiration') and asset_or_cash == 'Cash': # 9 10
            px = b2 - b4
        elif barrier_in_or_out == 'Out' and option_type in ('At-hit', 'At-expiration') and asset_or_cash == 'Asset': # 11 12
            px = a2 - a4
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'In' and option_type == 'Call' and asset_or_cash == 'Cash':  # 13
            px = b3 if strike_px > barrier_px else b1 - b2 + b4
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'In' and option_type == 'Call' and asset_or_cash == 'Cash':  # 14
            px = b1 if strike_px > barrier_px else b2 - b3 + b4
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'In' and option_type == 'Call' and asset_or_cash == 'Asset':  # 15
            px = a3 if strike_px > barrier_px else a1 - a2 + a4
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'In' and option_type == 'Call' and asset_or_cash == 'Asset':  # 16
            px = a1 if strike_px > barrier_px else a2 - a3 + a4
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'In' and option_type == 'Put' and asset_or_cash == 'Cash':  # 17
            px = b2 - b3 + b4 if strike_px > barrier_px else b1
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'In' and option_type == 'Put' and asset_or_cash == 'Cash':  # 18
            px = b1 - b2 + b4 if strike_px > barrier_px else b3
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'In' and option_type == 'Put' and asset_or_cash == 'Asset':  # 19
            px = a2 - a3 + a4 if strike_px > barrier_px else a1
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'In' and option_type == 'Put' and asset_or_cash == 'Asset':  # 20
            px = a1 - a2 + a3 if strike_px > barrier_px else a3
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'Out' and option_type == 'Call' and asset_or_cash == 'Cash':  # 21
            px = b1 - b3 if strike_px > barrier_px else b2 - b4
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'Out' and option_type == 'Call' and asset_or_cash == 'Cash':  # 22
            px = 0 if strike_px > barrier_px else b1 - b2 + b3 - b4
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'Out' and option_type == 'Call' and asset_or_cash == 'Asset':  # 23
            px = a1 - a3 if strike_px > barrier_px else a2 - a4
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'Out' and option_type == 'Call' and asset_or_cash == 'Asset':  # 24
            px = 0 if strike_px > barrier_px else a1 - a2 + a3 - a4
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'Out' and option_type == 'Put' and asset_or_cash == 'Cash':  # 25
            px = b1 - b2 + b3 - b4 if strike_px > barrier_px else 0
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'Out' and option_type == 'Put' and asset_or_cash == 'Cash':  # 26
            px = b2 - b4 if strike_px > barrier_px else b1 - b3
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'Out' and option_type == 'Put' and asset_or_cash == 'Asset':  # 27
            px = a1 - a2 + a3 - a4 if strike_px > barrier_px else 0
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'Out' and option_type == 'Put' and asset_or_cash == 'Asset':  # 28
            px = a2 - a4 if strike_px > barrier_px else a1 - a3
        else:
            raise ValueError
        return px

    # def checkEvent(self, pxs:(list, float, int)):
    #     """
    #     根据价格/价格序列，判断是否敲入/敲出,不修改self
    #     :param pxs: float, int, list[float]
    #     :return: event
    #     """
    #     # if isinstance(pxs, (int, float)):
    #     # BarrierEventEnum.KnockOut / BarrierEventEnum.KnockIn
    #     if isinstance(pxs, (float, int)):
    #         pxs = [pxs]
    #     event = self.event
    #     if not event or BarrierEventDict[event] == BarrierEventEnum.NoEvent:  # event == None   todo
    #         if self.barrierType == BarrierTypeEnum.UpIn:
    #             event = BarrierEventEnum.KnockIn if max(pxs) >= self.barrier_px else BarrierEventEnum.NoEvent
    #         elif self.barrierType == BarrierTypeEnum.UpOut:
    #             event = BarrierEventEnum.KnockOut if max(pxs) >= self.barrier_px else BarrierEventEnum.NoEvent
    #         elif self.barrierType == BarrierTypeEnum.DownIn:
    #             event = BarrierEventEnum.KnockIn if min(pxs) <= self.barrier_px else BarrierEventEnum.NoEvent
    #         elif self.barrierType == BarrierTypeEnum.DownOut:
    #             event = BarrierEventEnum.KnockOut if min(pxs) <= self.barrier_px else BarrierEventEnum.NoEvent
    #     return event

    def _get_factors(self, spot_px, strike_px, barrier_px, years_rf, option_type, barrier_up_or_down, barrier_in_or_out,
                     rf_rate, dvd_rate, volatility, years_vola=None, rebate_px=0):
        years_vola = years_vola if years_vola is not None else years_rf

        eta = 1 if barrier_up_or_down == 'Down' else -1
        if option_type == 'Call':
            phi = 1
        elif option_type == 'Put':
            phi = -1
        elif barrier_in_or_out == 'In':
            phi = -eta
        elif barrier_in_or_out == 'Out':
            phi = eta
        miu = (rf_rate - dvd_rate) / np.power(volatility, 2) - 0.5
        lambd = np.sqrt(np.power(miu, 2) + 2 * rf_rate / np.power(volatility, 2))
        sigma_sqt = volatility * np.sqrt(years_vola)

        x1 = np.log(spot_px / strike_px) / sigma_sqt + (1 + miu) * sigma_sqt
        x2 = np.log(spot_px / barrier_px) / sigma_sqt + (1 + miu) * sigma_sqt
        y1 = np.log(np.power(barrier_px, 2) / spot_px / strike_px) / sigma_sqt + (1 + miu) * sigma_sqt
        y2 = np.log(barrier_px / spot_px) / sigma_sqt + (1 + miu) * sigma_sqt
        z = np.log(barrier_px / spot_px) / sigma_sqt + lambd * sigma_sqt

        a1 = spot_px * np.exp(- dvd_rate * years_rf) * stats.norm.cdf(phi * x1)
        b1 = rebate_px * np.exp(- rf_rate * years_rf) * stats.norm.cdf(phi * (x1 - sigma_sqt))
        a2 = spot_px * np.exp(- dvd_rate * years_rf) * stats.norm.cdf(phi * x2)
        b2 = rebate_px * np.exp(- rf_rate * years_rf) * stats.norm.cdf(phi * (x2 - sigma_sqt))
        a3 = spot_px * np.exp(- dvd_rate * years_rf) * np.power(barrier_px / spot_px, 2 * (1 + miu)) * stats.norm.cdf(
            eta * y1)
        b3 = rebate_px * np.exp(- rf_rate * years_rf) * np.power(barrier_px / spot_px, 2 * miu) * stats.norm.cdf(
            eta * (y1 - sigma_sqt))
        a4 = spot_px * np.exp(- dvd_rate * years_rf) * np.power(barrier_px / spot_px, 2 * (1 + miu)) * stats.norm.cdf(
            eta * y2)
        b4 = rebate_px * np.exp(- rf_rate * years_rf) * np.power(barrier_px / spot_px, 2 * miu) * stats.norm.cdf(
            eta * (y2 - sigma_sqt))
        a5 = rebate_px * (np.power(barrier_px / spot_px, miu + lambd) *
                          stats.norm.cdf(eta * z) + stats.norm.cdf(eta * (z - 2 * lambd * sigma_sqt)) *
                          np.power(barrier_px / spot_px, miu - lambd))
        # print(eta, a1, b1, a2, b2, a3, b3, a4, b4, a5)
        return a1, b1, a2, b2, a3, b3, a4, b4, a5


def options_px():
    """The Complete Guide to Option"""
    price_X102, price_X98, s_lt = [], [], []
    H, T, r, d, b, sigma, K = 100, 0.5, 0.1, 0, 0.1, 0.2, 15
    for io in ['In', 'Out']:
        for ot in ['At-hit', 'At-expiration', 'Call', 'Put']:
            for at in ['Cash', 'Asset']:
                for ut in ['Down', 'Up']:
                    s = 105 if ut == 'Down' else 95
                    s_lt.append(s)
                    option = BinaryBarrierOptionAnalytical(option_type=ot, barrier_up_or_down=ut,
                                                           barrier_in_or_out=io, asset_or_cash=at,
                                                           spot_px=s, rebate_px=K, years_rf=T, rf_rate=r,
                                                           dvd_rate=d,
                                                           strike_px=102, barrier_px=H, volatility=sigma)
                    price_X102.append(option.get_px())
                    option = BinaryBarrierOptionAnalytical(option_type=ot, barrier_up_or_down=ut,
                                                           barrier_in_or_out=io, asset_or_cash=at,
                                                           spot_px=s, rebate_px=K, years_rf=T, rf_rate=r,
                                                           dvd_rate=d,
                                                           strike_px=98, barrier_px=H, volatility=sigma)
                    price_X98.append(option.get_px())
    print(
        pd.DataFrame(index=['S', 'X=102', 'X=98'],
                     columns=pd.MultiIndex.from_product([['In', 'Out'], ['At-hit', 'At-expiration', 'Call', 'Put'], ['Cash', 'Asset'],['Down', 'Up']]),
                     data=[s_lt, price_X102, price_X98]).T)

options_px()