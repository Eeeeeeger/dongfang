import warnings
from collections import OrderedDict
from pathlib import Path
from loguru import logger

warnings.filterwarnings('ignore')
# 绘制图形
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pyfolio as pf

plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('bmh')
colors = ['lightseagreen', 'lightcoral', 'slategrey']

class DataGenerator:
    def __init__(self, data_path: str= './input/macroFactorDf20230202_1.csv'):
        factors = pd.read_csv(data_path, index_col=0, parse_dates=True)
        targets = pd.read_csv('./input/index.csv', index_col=0, parse_dates=True)
        data = pd.concat([factors, targets], axis=1)
        data = pd.concat([data.iloc[:-1].dropna(), data.iloc[-1:]], axis=0)
        # self.assets = targets.columns.tolist()
        self.assets = ['000905.SH', 'NH0100.NHF', 'Bond']
        self.labels_dict = dict(zip(self.assets, [-1, 0, 1]))
        self.assets_dict = dict(zip([-1,0,1], self.assets))
        self.macro = factors.columns.tolist()

        # merge new features by first differencing
        self.data = data.join(data[self.macro].diff().rename(columns=dict(zip(self.macro, map(lambda x: f'd_{x}', self.macro)))))
        self.features = self.macro + list(map(lambda x: f'd_{x}', self.macro))
        self.data_dict = {'input': self.data.copy(), 'ret1d': self.data[self.assets].copy(), 'ret3d': self.data[self.assets].copy(),
                     'ret5d': self.data[self.assets].copy(),
                     'ret20d': self.data[self.assets].copy(),
                     'sign1d': self.data[self.assets].copy(), 'sign3d': self.data[self.assets].copy(), 'sign5d': self.data[self.assets].copy(),
                     'sign20d': self.data[self.assets].copy()}

        self.data.dropna(subset=self.features, how='any', inplace=True)
        self.dates = data.index.tolist()

    def create_labels(self):
        for asset in self.assets:
            self.data_dict['ret1d'][asset] = self.data_dict['input'][asset].pct_change(1).shift(-1)
            self.data_dict['ret3d'][asset] = self.data_dict['input'][asset].pct_change(3).shift(-3)
            self.data_dict['ret5d'][asset] = self.data_dict['input'][asset].pct_change(5).shift(-5)
            self.data_dict['ret20d'][asset] = self.data_dict['input'][asset].pct_change(20).shift(-20)
            self.data_dict['sign1d'][asset] = np.sign(self.data_dict['ret1d'][asset])
            self.data_dict['sign3d'][asset] = np.sign(self.data_dict['ret3d'][asset])
            self.data_dict['sign5d'][asset] = np.sign(self.data_dict['ret5d'][asset])
            self.data_dict['sign20d'][asset] = np.sign(self.data_dict['ret20d'][asset])

        # padding the index
        for k, v in self.data_dict.items():
            self.data_dict[k] = v.loc[v.index.intersection(self.data.index)]

    def create_relative_labels(self):
        for asset in self.assets:
            self.data_dict['ret1d'][asset] = self.data_dict['input'][asset].pct_change(1).shift(-1)
            self.data_dict['ret3d'][asset] = self.data_dict['input'][asset].pct_change(3).shift(-3)
            self.data_dict['ret5d'][asset] = self.data_dict['input'][asset].pct_change(5).shift(-5)
            self.data_dict['ret20d'][asset] = self.data_dict['input'][asset].pct_change(20).shift(-20)
        self.data_dict['sign1d'] = self.data_dict['ret1d'].idxmax(axis=1).apply(lambda x: self.labels_dict.get(x, np.nan))
        self.data_dict['sign3d'] = self.data_dict['ret3d'].idxmax(axis=1).apply(lambda x: self.labels_dict.get(x, np.nan))
        self.data_dict['sign5d'] = self.data_dict['ret5d'].idxmax(axis=1).apply(lambda x: self.labels_dict.get(x, np.nan))
        self.data_dict['sign20d'] = self.data_dict['ret20d'].idxmax(axis=1).apply(lambda x: self.labels_dict.get(x, np.nan))

        # padding the index
        for k, v in self.data_dict.items():
            self.data_dict[k] = v.loc[v.index.intersection(self.data.index)]


def daily_ret_statistic(factor_ret: pd.Series) -> pd.DataFrame:
    ################################# performance ##################################
    # 计算回撤
    factor_ret.index = pd.to_datetime(factor_ret.index)
    factor_net_value = (factor_ret).cumsum()
    drawdown = (factor_net_value.groupby(factor_net_value.index.year).cummax() - factor_net_value)  # 绝对值计算
    max_drawdown = drawdown.groupby(drawdown.index.year).max()
    drawdown_all = (factor_net_value.cummax() - factor_net_value)  # 绝对值计算
    max_drawdown_all = drawdown_all.max()
    # 计算年化夏普比率
    sharpe = (factor_ret.groupby(factor_ret.index.year).mean() / factor_ret.groupby(
        factor_ret.index.year).std() * np.sqrt(252))
    if factor_ret.std() != 0:
        sharpe_all = (factor_ret.mean() / factor_ret.std() * np.sqrt(252))
    else:
        sharpe_all = np.nan
    # 年化收益率
    annual_ret = (factor_ret.groupby(factor_ret.index.year).sum())
    annual_ret_all = (factor_ret.sum() / len(factor_ret) * 252)

    # calmar
    calmar = (annual_ret / max_drawdown)
    calmar_all = annual_ret_all / max_drawdown_all

    summary = pd.concat(
        [sharpe, annual_ret, max_drawdown, calmar],
        axis=1)
    summary.columns = ['sharpe', 'returns', 'drawdown', 'calmar']
    summary_all = pd.DataFrame([[sharpe_all, annual_ret_all, max_drawdown_all, calmar_all]])
    summary_all.columns = summary.columns
    summary_all.index = ['all']
    summary = pd.concat([summary, summary_all])

    # 处理小数点
    return summary.round(3)


def net_value_plot(ans: pd.DataFrame, result: pd.Series, data_dict: dict, assets: list, *args, **kwargs):
    pnl = result.dropna()
    # 计算累计收益
    cum = (pnl + 1).cumprod()
    # 计算回撤序列
    max_return = cum.cummax()
    drawdown = (cum - max_return) / max_return
    # 计算收益评价指标
    # 按年统计收益指标
    perf_stats_year = pnl.groupby(pnl.index.to_period('y')).apply(
        lambda data: pf.timeseries.perf_stats(data)).unstack()
    # 统计所有时间段的收益指标
    perf_stats_all = pf.timeseries.perf_stats((pnl)).to_frame(name='all')
    perf_stats = pd.concat([perf_stats_year, perf_stats_all.T], axis=0)
    perf_stats_ = round(perf_stats, 4).reset_index()

    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.5, 4]}, figsize=(20, 12))
    cols_names = ['date', 'Annual\nreturn', 'Cumulative\nreturns', 'Annual\nvolatility',
                  'Sharpe\nratio', 'Calmar\nratio', 'Stability', 'Max\ndrawdown',
                  'Omega\nratio', 'Sortino\nratio', 'Skew', 'Kurtosis', 'Tail\nratio',
                  'Daily value\nat risk']

    # 绘制表格
    ax0.set_axis_off()  # 除去坐标轴
    table = ax0.table(cellText=perf_stats_.values,
                      bbox=(0, 0, 1, 1),  # 设置表格位置， (x0, y0, width, height)
                      rowLoc='right',  # 行标题居中
                      cellLoc='right',
                      colLabels=cols_names,  # 设置列标题
                      colLoc='right',  # 列标题居中
                      edges='open'  # 不显示表格边框
                      )
    table.set_fontsize(13)

    # 绘制累计收益曲线
    ax2 = ax1.twinx()
    ax1.yaxis.set_ticks_position('right')  # 将回撤曲线的 y 轴移至右侧
    ax2.yaxis.set_ticks_position('left')  # 将累计收益曲线的 y 轴移至左侧
    # 绘制回撤曲线
    drawdown.plot.area(ax=ax1, label='drawdown (right)', rot=0, alpha=0.3, fontsize=13, grid=False)
    asset_each_day = ans.idxmax(axis=1)
    if kwargs.get('benchmark', False):
        for ii, each in enumerate(ans.columns):
            benchmark = data_dict['ret1d'].loc[pnl.index][[each]]
            benchmark = (benchmark + 1).cumprod()
            benchmark.plot(ax=ax2, label='Index (left)', rot=0, alpha=0.3, fontsize=13, grid=False, color=colors[ii])
            # ax2.bar(x=asset_each_day[asset_each_day == each].index, bottom=cum.min(), height=cum.max()-cum.min(), width= 0.2, alpha=0.3, color=colors[ii])
    # 绘制累计收益曲线
    cum.plot(ax=ax2, color='#F1C40F', lw=3.0, label='cumret (left)', rot=0, fontsize=13, grid=False)
    v = np.min([(data_dict['ret1d'].loc[pnl.index] + 1).cumprod().min().min(), cum.min()])
    colors_dict = dict(zip(assets, colors))
    prev_asset = asset_each_day.iloc[0]
    start = 0
    for ii in range(1, len(asset_each_day)):
        if asset_each_day.iloc[ii] != prev_asset:
            pd.Series(index=cum.index[start:ii], data=v).plot(ax=ax2, color=colors_dict[prev_asset], lw=5.0, alpha=0.5,
                                                              rot=0, grid=False)
            start = ii
            prev_asset = asset_each_day.iloc[ii]
    pd.Series(index=cum.index[start:], data=v).plot(ax=ax2, color=colors_dict[prev_asset], lw=5.0, alpha=0.5, rot=0,
                                                    grid=False)

    # 不然 x 轴留有空白
    ax2.set_xbound(lower=cum.index.min(), upper=cum.index.max())
    # 主轴定位器：每 5 个月显示一个日期：根据具体天数来做排版
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(250))
    # 同时绘制双轴的图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h2, l2 = h2[:4], l2[:4]
    plt.legend(h1 + h2, l1 + l2, fontsize=12, loc='upper left', ncol=1)

    fig.tight_layout()  # 规整排版
    if kwargs.get('save', True):
        p = Path(kwargs.get('path', './output/'))
        p.mkdir(parents=True, exist_ok=True)
        plt.savefig(p / f'{kwargs.get("name", "sample")}.png')
    else:
        plt.show()