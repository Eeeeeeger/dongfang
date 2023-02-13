import backtrader as bt
import datetime

class FuturesPandasData(bt.feeds.PandasData):
    lines = ('pre_close', 'amount', 'vwap', 'pre_settle', 'settle') # 要添加的线
    # 设置 line 在数据源上的列位置
    params=(
        ('pre_close', -1),
        ('amount', -1),
        ('vwap', -1),
        ('pre_settle', -1),
        ('settle', -1),
    )

class FuturesCSVData(bt.feeds.GenericCSVData):
    params = (
    # ('fromdate', datetime.datetime(2019,1,2)),
    # ('todate', datetime.datetime(2021,1,28)),
    ('nullvalue', float('NaN')),
    ('dtformat', '%Y-%m-%d'),
    # ('tmformat', '%H:%M:%S'),
    ('datetime', 0),
    # ('time', -1),
    ('pre_close', 1),
    ('open', 2),
    ('high', 3),
    ('low', 4),
    ('close', 5),
    ('volume', 6),
    ('amount', 7),
    ('vwap', 8),
    ('pre_settle', 9),
    ('settle', 10),
    ('openinterest', 11)
)
