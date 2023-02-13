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
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('bmh')

'''
initialize data
'''
factors = pd.read_csv('./input/macroFactorDf20230202_1.csv', index_col=0, parse_dates=True)
targets = pd.read_csv('./input/index.csv', index_col=0, parse_dates=True)
data = pd.concat([factors, targets], axis=1)
data = pd.concat([data.iloc[:-1].dropna(), data.iloc[-1:]], axis=0)
assets = targets.columns.tolist()
assets = ['000905.SH', 'NH0100.NHF', 'Bond']
macro = factors.columns.tolist()
data_dict = {'input': data, 'ret1d': data[assets].copy(), 'ret3d': data[assets].copy(), 'ret5d': data[assets].copy(),
             'ret20d': data[assets].copy(),
             'sign1d': data[assets].copy(), 'sign3d': data[assets].copy(), 'sign5d': data[assets].copy(),
             'sign20d': data[assets].copy()}

data.dropna(subset=macro, how='any', inplace=True)
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


def net_value_plot(result: pd.Series, *args, **kwargs):
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

    if kwargs.get('benchmark', False):
        benchmark = data_dict['ret1d'].loc[pnl.index]
        benchmark = (benchmark + 1).cumprod()
        benchmark.plot(ax=ax2, label='Index (left)', rot=0, alpha=0.3, fontsize=13, grid=False)

    # 绘制累计收益曲线
    cum.plot(ax=ax2, color='#F1C40F', lw=3.0, label='cumret (left)', rot=0, fontsize=13, grid=False)
    # 不然 x 轴留有空白
    ax2.set_xbound(lower=cum.index.min(), upper=cum.index.max())
    # 主轴定位器：每 5 个月显示一个日期：根据具体天数来做排版
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(250))
    # 同时绘制双轴的图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, fontsize=12, loc='upper left', ncol=1)

    fig.tight_layout()  # 规整排版
    if kwargs.get('save', True):
        p = Path('./output/backtest/')
        p.mkdir(parents=True, exist_ok=True)
        plt.savefig(p / f'{kwargs.get("name", "sample")}.png')
    else:
        plt.show()


def _regression(train_data: pd.DataFrame, predict_data: pd.DataFrame, asset: str, indices: list, params: dict):
    train_data = train_data[[asset] + macro].dropna()
    x = np.array(train_data.loc[:, macro])
    y = np.array(train_data.loc[:, asset])
    valid_ratio = 0.3
    x_train = x[:int(len(x) * (1 - valid_ratio))]
    y_train = y[:int(len(x) * (1 - valid_ratio))]
    x_dev = x[int(len(x) * (1 - valid_ratio)):]
    y_dev = y[int(len(x) * (1 - valid_ratio)):]
    del train_data
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_dev = lgb.Dataset(x_dev, y_dev, reference=lgb_train)
    del x_train, y_train, x_dev, y_dev
    tree = lgb.train(params, lgb_train, num_boost_round=1000,
                     valid_sets=lgb_dev, early_stopping_rounds=30)

    return pd.DataFrame(OrderedDict({asset: tree.predict(predict_data)}), index=predict_data.index).reindex(
        indices).ffill()


'''
params setting
'''

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    # 'objective': 'fair',
    'metric': 'l2',
    'max_depth': 4,
    'num_leaves': 16,
    'num_iterations': 200,
    'learning_rate': 0.001,
    'feature_fraction': 1,
    'bagging_fraction': 1,
    'bagging_freq': 5,
    'verbose': 1,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'device': 'cpu',
    'random_state': 100,
}

acc_lt, bt_lt = [], []
label_day_lt, p1_lt, p2_lt = [1, 3, 5, 20], [100, 200, 400], [0.0002, 0.0005, 0.001, 0.002]
train_data_dict = {}
valid_data_dict = {}
test_data_dict = {}
for k, v in data_dict.items():
    train_data_dict[k], valid_data_dict[k], test_data_dict[k] = v.iloc[:int(len(v) * 0.6)], v.iloc[
                                                                                            int(len(v) * 0.6):int(
                                                                                                len(v) * 0.8)], v.iloc[
                                                                                                                int(len(
                                                                                                                    v) * 0.8):]

for p1 in p1_lt:
    params['num_iterations'] = p1
    for p2 in p2_lt:
        params['learning_rate'] = p2
        for label_day in label_day_lt:
            label_type = f'ret{label_day}d'
            model = 'lgbRegressor'

            logger.info(
                f'label_day: {label_day}, model: {model}, leaves: {p1}, depths: {p2}\n'
                f'start train from {train_data_dict[label_type].index[0]} to {train_data_dict[label_type].index[-1]}')

            reg_data = pd.concat([train_data_dict[label_type][assets], train_data_dict['input'][macro]],
                                 axis=1)
            valid_data = valid_data_dict['input'][macro]
            indices = valid_data.index
            valid_data = valid_data.iloc[::label_day]
            valid_res = Parallel(n_jobs=len(assets))(
                delayed(_regression)(
                    reg_data,
                    valid_data,
                    asset,
                    indices,
                    params,
                )
                for asset in assets
            )

            valid_res = pd.concat(valid_res, axis=1)

            logger.info('start evaluate')
            ## 回测表现
            factor_ret = pd.Series(index=valid_res.index, name='ret', data=np.diagonal(
                valid_res.idxmax(axis=1).apply(lambda x: data_dict['ret1d'].loc[:, x]).loc[:,
                valid_res.index.tolist()])).dropna()
            summary = daily_ret_statistic(factor_ret)
            bt_lt.append(summary.iloc[-1])
            net_value_plot(factor_ret, benchmark=True, save=True,
                           name=f"valid_{label_day}_{model}_{p1}_{p2}")
    ## 预测准确率
    # sign = train_data_dict['sign1d'].copy().loc[:, assets]
    # acc_rate = (np.sign(valid_res) == np.sign(sign).loc[sign.index.intersection(valid_res.index)]).sum() / len(ans)
    # acc_lt.append(acc_rate)

bt_res = pd.concat(bt_lt, axis=1)
bt_res.columns = pd.MultiIndex.from_product(
    [label_day_lt, p1_lt, p2_lt])
bt_res.T.to_csv(f'output/tree_bt.csv')
#
# bt_res = pd.read_csv(f"output/bt"
#                      f".csv", index_col=[0,1,2,3]).T
# for each in bt_res.T['sharpe'].nlargest(10).index.to_list():
#     label_day, model, train_period, rolling_period = each
#     if rolling_period < label_day:
#         logger.debug('label_day is greater than rolling_period, reset rolling_period = label_day')
#         rolling_period = label_day
#     train_data_dict = {}
#     test_data_dict = {}
#     for k, v in data_dict.items():
#         train_data_dict[k], test_data_dict[k] = v.iloc[:int(len(v) * 0.6)], v.iloc[
#                                                                             int(len(v) * 0.6) - train_period:]
#
#     if model == 'LinearRegression':
#         label_type = 'ret1d'
#         m = LinearRegression()
#     elif model == 'LogisticRegression':
#         label_type = 'sign1d'
#         m = LogisticRegression(class_weight='balanced')
#     else:
#         raise ValueError
#
#     logger.info(
#         f'label_day: {label_day}, model: {model}, train_period: {train_period},rolling_period: {rolling_period}\n '
#         f'start backtest on test set from {test_data_dict[label_type].index[train_period]} to {test_data_dict[label_type].index[-1]}')
#
#
#     ans = pd.DataFrame()
#     for predict_num in tqdm(range(train_period, len(test_data_dict['input']), rolling_period)):
#         reg_data = pd.concat([test_data_dict[label_type][assets], test_data_dict['input'][macro]], axis=1).iloc[
#                    predict_num - train_period:predict_num]
#         predict_data = test_data_dict['input'][macro].iloc[predict_num:predict_num + rolling_period]
#         indices = predict_data.index
#         predict_data = predict_data.iloc[::label_day]
#         temp = Parallel(n_jobs=len(assets))(
#             delayed(_regression)(
#                 reg_data,
#                 predict_data,
#                 asset,
#                 indices,
#                 m,
#             )
#             for asset in assets
#         )
#
#         temp = pd.concat(temp, axis=1)
#         if ans.empty:
#             ans = temp
#         else:
#             ans = pd.concat([ans, temp], axis=0)
#
#     ## 回测表现
#     logger.info('start evaluate on test set')
#     factor_ret = pd.Series(index=ans.index, name='ret', data=np.diagonal(
#         ans.idxmax(axis=1).apply(lambda x: data_dict['ret1d'].loc[:, x]).loc[:, ans.index.tolist()])).dropna()
#     summary = daily_ret_statistic(factor_ret)
#     summary.to_csv(f'output/test_{label_day}_{model}_{train_period}_{rolling_period}.csv')
#     net_value_plot(factor_ret, benchmark=True, save=True, name=f"test_{label_day}_{model}_{train_period}_{rolling_period}")
