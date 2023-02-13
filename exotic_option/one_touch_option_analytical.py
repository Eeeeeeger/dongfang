# encoding: utf-8

import numpy as np
import pandas as pd
from scipy import stats

class OneTouchOptionAnalytical():
    """
    one-touch option
    参考《The Complete Guide to Option》 p152
    """

    def __init__(self, option_type, barrier_up_or_down, barrier_in_or_out,
                 spot_px, strike_px, barrier_px, rebate, maturity_years,
                 rf_rate, volatility, dvd_rate: float = 0, **kwargs):
        """
        初始化
        :param option_type: 期权类型 ('Call', 'Put')
        :param barrier_up_or_down: ('Up', 'Down')
        :param barrier_in_or_out: ('In', 'Out')
        :param spot_px: 即期价格
        :param strike_px: 行权价格
        :param barrier_px: 障碍价格
        :param rebate: （结束价格超过行权价格后）收益金额, rebate
        :param maturity_years: 到期年数
        :param rf_rate: 无风险利率（连续复利）
        :param volatility:  年化波动率
        :param dvd_rate: 分红利率（连续复利）
        """
        if option_type not in ('Call', 'Put'):
            raise ValueError
        if barrier_up_or_down not in ('Up', 'Down'):
            raise ValueError
        if barrier_in_or_out not in ('In', 'Out'):
            raise ValueError

        self.option_type = option_type
        self.barrier_up_or_down = barrier_up_or_down
        self.barrier_in_or_out = barrier_in_or_out
        self.spot_px = spot_px
        self.strike_px = strike_px
        self.barrier_px = barrier_px
        self.rebate = rebate
        self.maturity_years = maturity_years
        self.rf_rate = rf_rate
        self.volatility = volatility
        self.dvd_rate = dvd_rate
        # 由于定义法计算theta时会改变rf的值，cost_of_carry也应随时改变，所以每次使用cost_of_carry时都应当调用该函数
        super(OneTouchOptionAnalytical, self).__init__(**kwargs)

    def get_px(self):
        spot_px = self.spot_px
        strike_px = self.strike_px
        barrier_px = self.barrier_px
        maturity_years = self.maturity_years
        option_type = self.option_type
        barrier_up_or_down = self.barrier_up_or_down
        barrier_in_or_out = self.barrier_in_or_out
        rf_rate = self.rf_rate
        dvd_rate = self.dvd_rate
        volatility = self.volatility
        rebate = self.rebate
        
        # if self.event == BarrierEventEnum.KnockOut.value and self.barrierType in (
        # BarrierTypeEnum.UpOut, BarrierTypeEnum.DownOut):
        #     return self.rebateAmount
        # elif self.event == BarrierEventEnum.KnockIn.value and self.barrierType in (
        # BarrierTypeEnum.UpIn, BarrierTypeEnum.DownIn):
        #     return VanillaEuropeanOption(self.optionType, ExerciseTypeEnum.European,
        #                                  spot_px, strike_px, self.maturity_years, rf_rate, dvd_rate,
        #                                  self.volatility).get_analytical_px()
        
        a, b, c, d, e, f = self._get_factors(spot_px, strike_px, barrier_px, maturity_years, option_type, barrier_up_or_down,
                     rf_rate, dvd_rate, volatility, rebate)

        if barrier_up_or_down == 'Down' and barrier_in_or_out == 'In' and option_type == 'Call':    # cdi
            px = c + e if strike_px > barrier_px else a - b + d + e
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'In' and option_type == 'Call':    # cui
            px = a + e if strike_px > barrier_px else b - c + d + e
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'In' and option_type == 'Put':    # pdi
            px = b - c + d + e if strike_px > barrier_px else a + e
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'In' and option_type == 'Put':    # pui
            px = a - b + d + e if strike_px > barrier_px else c + e
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'Out' and option_type == 'Call':    # cdo
            px = a - c + f if strike_px > barrier_px else b - d + f
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'Out' and option_type == 'Call':    # cuo
            px = f if strike_px > barrier_px else a - b + c - d + f
        elif barrier_up_or_down == 'Down' and barrier_in_or_out == 'Out' and option_type == 'Put':    # pdo
            px = a - b + c - d + f if strike_px > barrier_px else f
        elif barrier_up_or_down == 'Up' and barrier_in_or_out == 'Out' and option_type == 'Put':    # puo
            px = b - d + f if strike_px > barrier_px else a - c + f
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

    def _get_factors(self, spot_px, strike_px, barrier_px, maturity_years, option_type, barrier_up_or_down,
                     rf_rate, dvd_rate, volatility, rebate=0):
        phi = 1 if option_type == 'Call' else -1
        eta = 1 if barrier_up_or_down == 'Down' else -1  # todo

        miu = (rf_rate - dvd_rate) / np.power(volatility, 2) - 0.5
        lambd = np.sqrt(np.power(miu, 2) + 2 * rf_rate / np.power(volatility, 2))
        sigma_sqt = volatility * np.sqrt(maturity_years)

        x1 = np.log(spot_px / strike_px) / sigma_sqt + (1 + miu) * sigma_sqt
        x2 = np.log(spot_px / barrier_px) / sigma_sqt + (1 + miu) * sigma_sqt
        y1 = np.log(np.power(barrier_px, 2) / spot_px / strike_px) / sigma_sqt + (1 + miu) * sigma_sqt
        y2 = np.log(barrier_px / spot_px) / sigma_sqt + (1 + miu) * sigma_sqt
        z = np.log(barrier_px / spot_px) / sigma_sqt + lambd * sigma_sqt

        a = phi * spot_px * np.exp(- dvd_rate * maturity_years) * stats.norm.cdf(phi * x1) - \
                 phi * strike_px * np.exp(- rf_rate * maturity_years) * \
                 stats.norm.cdf(phi * (x1 - sigma_sqt))
        b = phi * spot_px * np.exp(- dvd_rate * maturity_years) * stats.norm.cdf(phi * x2) - \
                 phi * strike_px * np.exp(- rf_rate * maturity_years) * \
                 stats.norm.cdf(phi * (x2 - sigma_sqt))
        c = phi * spot_px * np.exp(- dvd_rate * maturity_years) * \
                 np.power(barrier_px / spot_px, 2 * (1 + miu)) * stats.norm.cdf(eta * y1) - \
                 phi * strike_px * np.exp(- rf_rate * maturity_years) * \
                 np.power(barrier_px / spot_px, 2 * miu) * stats.norm.cdf(eta * (y1 - sigma_sqt))
        d = phi * spot_px * np.exp(- dvd_rate * maturity_years) * \
                 np.power(barrier_px / spot_px, 2 * (1 + miu)) * stats.norm.cdf(eta * y2) - \
                 phi * strike_px * np.exp(- rf_rate * maturity_years) * \
                 np.power(barrier_px / spot_px, 2 * miu) * stats.norm.cdf(eta * (y2 - sigma_sqt))
        e = rebate * np.exp(- rf_rate * maturity_years) * \
                 (stats.norm.cdf(eta * (x2 - sigma_sqt)) - stats.norm.cdf(eta * (y2 - sigma_sqt)) *
                  np.power(barrier_px / spot_px, 2 * miu))
        f = rebate * (np.power(barrier_px / spot_px, miu + lambd) *
                  stats.norm.cdf(eta * z) + stats.norm.cdf(eta * (z - 2 * lambd * sigma_sqt)) *
                  np.power(barrier_px / spot_px, miu - lambd))
        return a, b, c, d, e, f

def options_px():
    """The Complete Guide to Option"""
    # P154
    price_vol25, price_vol30 = [], []
    for io in ['Out', 'In']:
        for ot in ['Call', 'Put']:
            for h in [95, 100]:
                for s in [90, 100, 110]:
                    option = OneTouchOptionAnalytical(option_type=ot, barrier_up_or_down='Down', barrier_in_or_out=io,
                                                      spot_px=100, rebate=3, maturity_years=0.5, rf_rate=0.08, dvd_rate=0.04,
                                                      strike_px=s, barrier_px=h, volatility=0.25)
                    price_vol25.append(option.get_px())
                    option = OneTouchOptionAnalytical(option_type=ot, barrier_up_or_down='Down', barrier_in_or_out=io,
                                                      spot_px=100, rebate=3, maturity_years=0.5, rf_rate=0.08, dvd_rate=0.04,
                                                      strike_px=s, barrier_px=h, volatility=0.3)
                    price_vol30.append(option.get_px())
    print(
          pd.DataFrame(index=['vol=0.25', 'vol=0.3'],
                 columns=pd.MultiIndex.from_product([['Out','In'],['Call', 'Put'],[95, 100],[90, 100, 110]]),
                 data=[price_vol25, price_vol30]).T)

    price_vol25, price_vol30 = [], []
    for io in ['Out', 'In']:
        for ot in ['Call', 'Put']:
            for h in [105]:
                for s in [90, 100, 110]:
                    option = OneTouchOptionAnalytical(option_type=ot, barrier_up_or_down='Up', barrier_in_or_out=io,
                                                      spot_px=100, rebate=3, maturity_years=0.5, rf_rate=0.08, dvd_rate=0.04,
                                                      strike_px=s, barrier_px=h, volatility=0.25)
                    price_vol25.append(option.get_px())
                    option = OneTouchOptionAnalytical(option_type=ot, barrier_up_or_down='Up', barrier_in_or_out=io,
                                                      spot_px=100, rebate=3, maturity_years=0.5, rf_rate=0.08, dvd_rate=0.04,
                                                      strike_px=s, barrier_px=h, volatility=0.3)
                    price_vol30.append(option.get_px())
    print(
          pd.DataFrame(index=['vol=0.25', 'vol=0.3'],
                 columns=pd.MultiIndex.from_product([['Out','In'],['Call', 'Put'],[90, 100, 110],[105]]),
                 data=[price_vol25, price_vol30]).T)

options_px()
