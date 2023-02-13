import warnings
from pathlib import Path

import backtrader as bt
import pandas as pd

from TradingUtils.dataflow import FuturesPandasData
from TradingUtils.strategy import FuturesRollingStrategy
from TradingUtils.visualize import net_value_plot
warnings.filterwarnings('ignore')

cerebro = bt.Cerebro()
cols = ['pre_close', 'open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'pre_settle', 'settle', 'openinterest','LTDATE_NEW']
lt = []
for x in Path('input/T/').glob('*.csv'):
    if '15' in str(x) or '16' in str(x) or '17' in str(x) or '18' in str(x):
        continue
    df = pd.read_csv(x, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = cols
    df = df.ffill()
    df['sec_code'] = x.stem
    lt.append(df)
daily_info = pd.concat(lt, axis=0)
date_list = daily_info.index.unique().sort_values()

for code in daily_info['sec_code'].unique():
    temp_df = pd.DataFrame(index=date_list)
    df = daily_info.query(f" sec_code == '{code}' ")[cols]
    df = pd.merge(temp_df, df, left_index=True, right_index=True, how='left').fillna(0)
    cerebro.adddata(FuturesPandasData(dataname=df), name=code)

cerebro.addstrategy(FuturesRollingStrategy, **{'reserve': 0.3,
                                               'signal_file': './input/signal/diff.csv'})

cerebro.broker.setcash(100000000.0)
cerebro.broker.setcommission(
    commission=3,
    # margin=0.02,
    automargin=0.02 * 10000,
    mult=10000,
    commtype=bt.comminfo.CommInfoBase.COMM_FIXED,
    stocklike=False,

)
# cerebro.broker.set_filler(bt.broker.fillers.FixedBarPerc(perc=50))

cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')

result = cerebro.run()

net_value_plot(result, **{'save': True, 'name': 'diff'})

# # 佣金，双边各 0.0003
# cerebro.broker.setcommission(commission=0.0003)
# # 滑点：双边各 0.0001
# cerebro.broker.set_slippage_perc(perc=0.0001)

# cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl') # 返回收益率时序数据
# cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn') # 年化收益率
# cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio') # 夏普比率
# cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown') # 回撤


# 拼接期货合约
# data0 = bt.feeds.MyFeed(dataname='Expiry0')
# data1 = bt.feeds.MyFeed(dataname='Expiry1')
# ...
# dataN = bt.feeds.MyFeed(dataname='ExpiryN')
#
# drollover = bt.feeds.RollOver(data0, data1, ..., dataN, dataname='MyRoll', **kwargs)
# cerebro.adddata(drollover)
#
# cerebro.run()

# for x in Path('./input/quote/').glob('*.csv'):
#     # TODO: date is not padding, bt.Strategy.__next__ will have bug
#     cerebro.adddata(FuturesCSVData(dataname=str(x)), name=x.stem)
