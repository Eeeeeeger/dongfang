import numpy as np
import pandas as pd
import importlib
from pathlib import Path
from typing import Optional, List, Union
import pyfolio as pf
from DominantContract import BaseContract, MovingContract
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # 导入设置坐标轴的模块
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.style.use('seaborn')  # plt.style.use('dark_background')


class BaseModel(object):
    def __init__(self,
                 dominant_type: str = 'AfterMovingContract',
                 trade_type: bool = False,
                 days: int = 3,
                 ):
        '''
        Args:
            dominant_type: 选择国债期货主力合成的方式 DominantContract模块里的具体类
            days: 国债期货主力合成的参数
        '''
        self.dominant_type = dominant_type
        self.trade_type = trade_type
        if dominant_type in dir(MovingContract):
            dominant_cls = getattr(MovingContract, dominant_type)()
        elif dominant_type in dir(BaseContract):
            dominant_cls = getattr(BaseContract, dominant_type)()
        else:
            raise ValueError('wrong input for `dominant_type`')
        # 国债期货连续合约拼接
        dominant_cls.concat(days=days)
        if trade_type:
            dominant_cls.trade()
            self.futures = dominant_cls.trade_data
        else:
            self.futures = dominant_cls.dominant_data.reset_index().set_index('date')
        self.futures.index = pd.to_datetime(self.futures.index)
        # 10年国债收益率
        self.bond = pd.read_csv(f'./input/BondYield_CCDC.csv', parse_dates=['date'], index_col=0)
        indices = self.bond.index.intersection(self.futures.index)
        self.bond = self.bond.loc[indices]
        self.futures = self.futures.loc[indices]
        logger.info(f'init {self.__class__.__name__}: {dominant_type}_days{days} with start_date: {self.bond.index[0].date()}, end_date: {self.bond.index[-1].date()}')

    def calc_hedge_ratio(self):
        pass

    def ajust_hedge_frequency(self):
        self.check_frequency()
        pass

    def hedge(self):
        self.calc_hedge_ratio()
        self.ajust_hedge_frequency()

    @staticmethod
    def check_frequency(frequency):
        if isinstance(frequency, str):
            if frequency not in ['daily', 'weekly', 'monthly', 'yearly']:
                raise ValueError('wrong input for `frequency`')
        if isinstance(frequency, int):
            if frequency < 1:
                raise ValueError('wrong input for `frequency`')

    # 统计指标及绘图
    @staticmethod
    def summary_info(daily_ret: pd.DataFrame, *args, **kwargs):
        logger.info('start summary and plot ')
        bond_ret = daily_ret['bond']
        bond_cumret = (daily_ret['bond'].fillna(0) + 1).cumprod()
        futures_ret = daily_ret['futures']
        futures_cumret = (daily_ret['futures'].fillna(0) + 1).cumprod()
        hedge_ret = daily_ret['hedge']
        hedge_cumret = (daily_ret['hedge'].fillna(0) + 1).cumprod()

        # 计算收益评价指标
        stats = []
        for ret in [bond_ret, futures_ret, hedge_ret]:
            perf_stats_year = ret.groupby(ret.index.to_period('y')).apply(
                lambda data: pf.timeseries.perf_stats(data)).unstack()
            # 统计所有时间段的收益指标
            perf_stats_all = pf.timeseries.perf_stats(ret).to_frame(name='all')
            perf_stats = pd.concat([perf_stats_year, perf_stats_all.T], axis=0)[['Annual return', 'Annual volatility', 'Sharpe ratio', 'Max drawdown']]
            perf_stats.columns = perf_stats.columns.map(lambda x: ret.name+' '+x.lower())
            stats.append(perf_stats)
        stats = pd.concat(stats, axis=1)
        stats['variance reduction'] = 1 - stats['hedge annual volatility'] / stats['bond annual volatility']
        stats = stats.applymap(lambda x: format(x, '.4f')).reset_index()

        fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.5, 4]}, figsize=(20, 12))
        # 绘制表格
        ax0.set_axis_off()  # 除去坐标轴
        table = ax0.table(cellText=stats.values,
                          bbox=(0, 0, 1, 1),  # 设置表格位置， (x0, y0, width, height)
                          rowLoc='right',  # 行标题居中
                          cellLoc='right',
                          colLabels=stats.columns,  # 设置列标题
                          colLoc='right',  # 列标题居中
                          edges='open'  # 不显示表格边框
                          )
        table.set_fontsize(15)

        # 绘制累计收益曲线
        ax2 = ax1.twinx()
        ax1.yaxis.set_ticks_position('right')  # 将回撤曲线的 y 轴移至右侧
        ax2.yaxis.set_ticks_position('left')  # 将累计收益曲线的 y 轴移至左侧
        (daily_ret['ratio']).abs().plot.area(ax=ax1, label='hedge ratio (right)', rot=0, alpha=0.3, fontsize=13, grid=False)
        # 绘制10年国债收益率
        (np.sqrt(250)*bond_ret.rolling(30).std()).plot(ax=ax2, label='Bond Vol(left)', rot=0, alpha=0.3, fontsize=13, grid=False)
        bond_cumret.plot(ax=ax2, label='Bond (left)', rot=0, alpha=0.3, fontsize=13, grid=False)
        # 绘制累计收益曲线
        futures_cumret.plot(ax=ax2, label='Futures (left)', rot=0, alpha=0.3, fontsize=13, grid=False)
        hedge_cumret.plot(ax=ax2, color='#F1C40F', lw=3.0, label='Hedge (left)', rot=0, fontsize=13, grid=False)
        ax2.set_xbound(lower=hedge_cumret.index.min(), upper=hedge_cumret.index.max())
        # 主轴定位器：每 5 个月显示一个日期：根据具体天数来做排版
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(250))
        # 同时绘制双轴的图例
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        plt.legend(h1 + h2, l1 + l2, fontsize=12, loc='upper left', ncol=1)
        fig.tight_layout()
        if kwargs.get('save', True):
            p = Path('./output/hedge')
            p.mkdir(parents=True, exist_ok=True)
            save_path = p / f'{kwargs.get("name","sample")}.png'
            plt.savefig(save_path)
            logger.info(f'end summary and plot with save path: {str(save_path)}')
        else:
            plt.show()
            logger.info(f'end summary and plot')

        return stats
