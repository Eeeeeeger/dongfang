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
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('bmh')
colors = ['lightseagreen', 'lightcoral', 'slategrey']

'''
initialize data
'''
factors = pd.read_csv('./input/macroFactorDf20230202_1.csv', index_col=0, parse_dates=True)
targets = pd.read_csv('./input/index.csv', index_col=0, parse_dates=True)
data = pd.concat([factors, targets], axis=1)
data = pd.concat([data.iloc[:-1].dropna(), data.iloc[-1:]], axis=0)
assets = targets.columns.tolist()
assets = ['000905.SH', 'NH0100.NHF', 'Bond']
labels_dict = dict(zip(assets,[-1,0,1]))
macro = factors.columns.tolist()
# new features
data = data.join(data[macro].diff().rename(columns=dict(zip(macro, map(lambda x: f'd_{x}', macro)))))
features = macro + list(map(lambda x: f'd_{x}', macro))
data_dict = {'input': data.copy(), 'ret1d': data[assets].copy(), 'ret3d': data[assets].copy(),
             'ret5d': data[assets].copy(),
             'ret20d': data[assets].copy(),
             'sign1d': data[assets].copy(), 'sign3d': data[assets].copy(), 'sign5d': data[assets].copy(),
             'sign20d': data[assets].copy()}

data.dropna(subset=features, how='any', inplace=True)
dates = data.index.tolist()

for asset in assets:
    data_dict['ret1d'][asset] = data_dict['input'][asset].pct_change(1).shift(-1)
    data_dict['ret3d'][asset] = data_dict['input'][asset].pct_change(3).shift(-3)
    data_dict['ret5d'][asset] = data_dict['input'][asset].pct_change(5).shift(-5)
    data_dict['ret20d'][asset] = data_dict['input'][asset].pct_change(20).shift(-20)
    data_dict['sign1d'][asset] = np.sign(data_dict['ret1d'][asset])
    data_dict['sign3d'][asset] = np.sign(data_dict['ret3d'][asset])
    data_dict['sign5d'][asset] = np.sign(data_dict['ret5d'][asset])
    data_dict['sign20d'][asset] = np.sign(data_dict['ret20d'][asset])

for k, v in data_dict.items():
    data_dict[k] = v.loc[v.index.intersection(data.index)]
del data


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


def net_value_plot(ans: pd.DataFrame, result: pd.Series, *args, **kwargs):
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
        p = Path('./output/regression/backtest/')
        p.mkdir(parents=True, exist_ok=True)
        plt.savefig(p / f'{kwargs.get("name", "sample")}.png')
    else:
        plt.show()

def _regression(train_data: pd.DataFrame, predict_data: pd.DataFrame, asset: str, indices: list,
                model: LinearRegression or LogisticRegression):
    train_data = train_data[[asset] + features].dropna()

    ss = StandardScaler()
    train_data.loc[:, features] = ss.fit_transform(train_data.loc[:, features])
    predict_data.loc[:, features] = ss.transform(predict_data.loc[:, features])
    lr = model.fit(X=train_data.loc[:, features], y=train_data.loc[:, asset])
    if isinstance(model, LinearRegression):
        return pd.DataFrame(OrderedDict({asset: lr.predict(predict_data)}), index=predict_data.index).reindex(
            indices).ffill()
    elif isinstance(model, LogisticRegression):
        return pd.DataFrame(OrderedDict({asset: lr.predict_proba(predict_data)[:, 1]}),
                            index=predict_data.index).reindex(indices).ffill()


'''
params setting
'''
#
# acc_lt, bt_lt = [], []
#
# for label_day in [1, 3, 5, 20]:
#     for model in ["LinearRegression", "LogisticRegression"]:
#
#         if model == 'LinearRegression':
#             label_type = f'ret{label_day}d'
#             m = LinearRegression()
#         elif model == 'LogisticRegression':
#             label_type = f'sign{label_day}d'
#             m = LogisticRegression(class_weight='balanced')
#         else:
#             raise ValueError
#
#         for train_period in [120, 250, 500, 750]:
#             train_data_dict = {}
#             test_data_dict = {}
#             for k, v in data_dict.items():
#                 train_data_dict[k], test_data_dict[k] = v.iloc[:int(len(v) * 0.7)], v.iloc[
#                                                                                     int(len(v) * 0.7) - train_period:]
#
#             for rolling_period in [1, 20, 60, 120]:
#                 if rolling_period < label_day:
#                     logger.debug('label_day is greater than rolling_period, reset rolling_period = label_day')
#                     rolling_period = label_day
#                 # train_period = 250
#                 # rolling_period = 100
#                 ans = pd.DataFrame()
#                 logger.info(
#                     f'label_day: {label_day}, model: {model}, train_period: {train_period},rolling_period: {rolling_period}\n'
#                     f'start backtest on train set from {train_data_dict[label_type].index[train_period]} to {train_data_dict[label_type].index[-1]}')
#                 for predict_num in tqdm(range(train_period, len(train_data_dict['input']), rolling_period)):
#                     reg_data = pd.concat([train_data_dict[label_type][assets], train_data_dict['input'][features]],
#                                          axis=1).iloc[
#                                predict_num - train_period:predict_num]
#                     predict_data = train_data_dict['input'][features].iloc[predict_num:predict_num + rolling_period]
#                     indices = predict_data.index
#
#                     predict_data = predict_data.iloc[::label_day]
#                     temp = Parallel(n_jobs=len(assets))(
#                         delayed(_regression)(
#                             reg_data,
#                             predict_data,
#                             asset,
#                             indices,
#                             m,
#                         )
#                         for asset in assets
#                     )
#
#                     temp = pd.concat(temp, axis=1)
#                     if ans.empty:
#                         ans = temp
#                     else:
#                         ans = pd.concat([ans, temp], axis=0)
#
#                 logger.info('start evaluate on train set')
#                 ## 回测表现
#                 factor_ret = pd.Series(index=ans.index, name='ret', data=np.diagonal(
#                     ans.idxmax(axis=1).apply(lambda x: data_dict['ret1d'].loc[:, x]).loc[:,
#                     ans.index.tolist()])).dropna()
#                 summary = daily_ret_statistic(factor_ret)
#                 bt_lt.append(summary.iloc[-1])
#                 net_value_plot(ans, factor_ret, benchmark=True, save=True,
#                                name=f"{label_day}_{model}_{train_period}_{rolling_period}")
#                 ## 预测准确率
#                 sign = train_data_dict['sign1d'].copy().loc[:, assets]
#                 acc_rate = (np.sign(ans) == np.sign(sign).loc[sign.index.intersection(ans.index)]).sum() / len(ans)
#                 acc_lt.append(acc_rate)
#
# acc_res, bt_res = pd.concat(acc_lt, axis=1), pd.concat(bt_lt, axis=1)
# indices = pd.MultiIndex.from_product(
#     [[1, 3, 5, 20], ['LinearRegression', 'LogisticRegression'], [120, 250, 500, 750], [1, 20, 60, 120]])
# acc_res.columns, bt_res.columns = indices, indices
# acc_res.loc['mean'] = acc_res.mean()
# acc_res.T.to_csv(f'output/regression/acc.csv')
# bt_res.T.to_csv(f'output/regression/bt.csv')

bt_res = pd.read_csv(f"output/regression/bt.csv", index_col=[0, 1, 2, 3]).T
for each in bt_res.T['sharpe'].nlargest(5).index.to_list():
    print(each)
    label_day, model, train_period, rolling_period = each
    if rolling_period < label_day:
        logger.debug('label_day is greater than rolling_period, reset rolling_period = label_day')
        rolling_period = label_day
    train_data_dict = {}
    test_data_dict = {}
    for k, v in data_dict.items():
        train_data_dict[k], test_data_dict[k] = v.iloc[:int(len(v) * 0.7)], v.iloc[
                                                                            int(len(v) * 0.7) - train_period:]

    if model == 'LinearRegression':
        label_type = f'ret{label_day}d'
        m = LinearRegression()
    elif model == 'LogisticRegression':
        label_type = f'sign{label_day}d'
        m = LogisticRegression(class_weight='balanced')
    else:
        raise ValueError

    logger.info(
        f'label_day: {label_day}, model: {model}, train_period: {train_period},rolling_period: {rolling_period}\n '
        f'start backtest on test set from {test_data_dict[label_type].index[train_period]} to {test_data_dict[label_type].index[-1]}')

    ans = pd.DataFrame()
    for predict_num in tqdm(range(train_period, len(test_data_dict['input']), rolling_period)):
        reg_data = pd.concat([test_data_dict[label_type][assets], test_data_dict['input'][features]], axis=1).iloc[
                   predict_num - train_period:predict_num]
        predict_data = test_data_dict['input'][features].iloc[predict_num:predict_num + rolling_period]
        indices = predict_data.index
        predict_data = predict_data.iloc[::label_day]
        temp = Parallel(n_jobs=len(assets))(
            delayed(_regression)(
                reg_data,
                predict_data,
                asset,
                indices,
                m,
            )
            for asset in assets
        )

        temp = pd.concat(temp, axis=1)
        if ans.empty:
            ans = temp
        else:
            ans = pd.concat([ans, temp], axis=0)

    ## 回测表现
    logger.info('start evaluate on test set')
    factor_ret = pd.Series(index=ans.index, name='ret', data=np.diagonal(
        ans.idxmax(axis=1).apply(lambda x: data_dict['ret1d'].loc[:, x]).loc[:, ans.index.tolist()])).dropna()
    summary = daily_ret_statistic(factor_ret)
    summary.to_csv(f'output/regression/test_{label_day}_{model}_{train_period}_{rolling_period}.csv')
    net_value_plot(ans, factor_ret, benchmark=True, save=True,
                   name=f"test_{label_day}_{model}_{train_period}_{rolling_period}")
