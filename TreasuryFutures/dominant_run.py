import warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from DominantContract.BaseContract import DirectContract, BeforeContract, AfterContract
from DominantContract.MovingContract import AfterMovingContract, BeforeMovingContract
import numpy as np
from sklearn.metrics import mean_squared_error
from DynamicHedge.StatisticsModel import OLSModel, WLSModel,VARModel
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.style.use('seaborn')  # plt.style.use('dark_background')
warnings.filterwarnings('ignore')

#
# freq_lt = ['monthly', 'daily', 5, 10, 20]
# trace_lt = [20, 40, 60, 80, 100]
# train, valid, test = [],[],[]
# hedge_cls = VARModel()
# for freq in freq_lt:
#     for trace in trace_lt:
#         res = hedge_cls.hedge(frequency=freq, trace_back=trace).set_index('index')
#         train.append(res['variance reduction'].iloc[:4].astype(np.float64).mean())
#         valid.append(res['variance reduction'].iloc[4:6].astype(np.float64).mean())
#         test.append(res['variance reduction'].iloc[6:8].astype(np.float64).mean())
#
# df = pd.DataFrame({'train': train, 'valid': valid, 'test': test}, index=pd.MultiIndex.from_product([freq_lt, trace_lt]))
#
#

days = 5
# 合成主力价格序列
a = AfterMovingContract()
a.concat(days=3)
a.trade()

futures = a.dominant_data.reset_index().set_index('date')
futures.index = pd.to_datetime(futures.index)
futures.reset_index().groupby('code')['date'].agg(['first', 'last'])
# 读取十年收益率数据
bond = pd.read_csv('./input/BondYield_CCDC.csv', parse_dates=['date'], index_col=0)
# 对齐日期
indices = bond.index.intersection(futures.index)
bond = bond.loc[indices]
futures = futures.loc[indices]

futures_ret, futures_cumret = futures['ret'], futures['cum_ret']
bond_ret, bond_cumret = bond['ret'], bond['cum_ret']
# 统计信息
perf_stats_ = pd.DataFrame(columns=['year', 'days', 'ret_corr', 'cumret_corr', 'ret_rmse', 'cumret_rmse'])
perf_stats_['year'] = indices.year.unique()
perf_stats_['days'] = bond_ret.groupby(bond_ret.index.year).count().tolist()
perf_stats_['ret_corr'] = bond_ret.groupby(bond_ret.index.year).corr(futures_ret).tolist()
perf_stats_['cumret_corr'] = bond_cumret.groupby(bond_cumret.index.year).corr(futures_cumret).tolist()
for ii, y in enumerate(perf_stats_['year'].tolist()):
    perf_stats_['ret_rmse'].iloc[ii] = np.sqrt(
        mean_squared_error(bond_ret[bond_ret.index.year == y], futures_ret[futures_ret.index.year == y]))
    perf_stats_['cumret_rmse'].iloc[ii] = np.sqrt(
        mean_squared_error(bond_cumret[bond_cumret.index.year == y], futures_cumret[futures_cumret.index.year == y]))

perf_stats_ = perf_stats_.append(
    pd.DataFrame(
        {'year': 'all', 'days': len(bond_ret),
        'ret_corr': bond_ret.corr(futures_ret), 'cumret_corr': bond_cumret.corr(futures_cumret),
        'ret_rmse': np.sqrt(mean_squared_error(bond_ret, futures_ret)), 'cumret_rmse': np.sqrt(mean_squared_error(bond_cumret, futures_cumret))
        },
        index=[0]
    )
)
perf_stats_ = round(perf_stats_, 6)
cols_names = perf_stats_.columns.to_list()

fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.5, 4]}, figsize=(20, 12))

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

bond_cumret.plot(ax=ax1, label='10YBondYield', rot=0, alpha=0.3, fontsize=13, grid=False)
futures_cumret.plot(ax=ax1, label=a.__class__.__name__, rot=0, alpha=0.3, fontsize=13, grid=False)
plt.legend(fontsize=12, loc='upper left', ncol=1)
p = Path('./output/dominant/')
p.mkdir(parents=True, exist_ok=True)
plt.savefig(p / f'{a.__class__.__name__}_{days}.png')
