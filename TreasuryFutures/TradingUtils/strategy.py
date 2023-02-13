import backtrader as bt
import pandas as pd
from typing import Optional
from loguru import logger

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.count = 0  # 用于计算 next 的循环次数
        # 打印数据集和数据集对应的名称
        print("------------- init 中的索引位置-------------")
        print(self.datas[0].lines.getlinealiases())
        print("0 索引：", 'datetime', self.data.lines.datetime.date(0), 'close', self.data.lines.close[0])
        print("-1 索引：", 'datetime', self.data.lines.datetime.date(-1), 'close', self.data.lines.close[-1])
        print("-2 索引", 'datetime', self.data.lines.datetime.date(-2), 'close', self.data.lines.close[-2])
        print("1 索引：", 'datetime', self.data.lines.datetime.date(1), 'close', self.data.lines.close[1])
        print("2 索引", 'datetime', self.data.lines.datetime.date(2), 'close', self.data.lines.close[2])
        print("从 0 开始往前取3天的收盘价：", self.data.lines.close.get(ago=0, size=3))
        print("从-1开始往前取3天的收盘价：", self.data.lines.close.get(ago=-1, size=3))
        print("从-2开始往前取3天的收盘价：", self.data.lines.close.get(ago=-2, size=3))
        print("line的总长度：", self.data.buflen())

    def next(self):
        print(f"------------- next 的第{self.count + 1}次循环 --------------")
        print("当前时点（今日）：", 'datetime', self.data.lines.datetime.date(0), 'close', self.data.lines.close[0])
        print("往前推1天（昨日）：", 'datetime', self.data.lines.datetime.date(-1), 'close', self.data.lines.close[-1])
        print("往前推2天（前日）", 'datetime', self.data.lines.datetime.date(-2), 'close', self.data.lines.close[-2])
        # print("前日、昨日、今日的收盘价：", self.data.lines.close.get(ago=0, size=3))
        # print("往后推1天（明日）：", 'datetime', self.data.lines.datetime.date(1), 'close', self.data.lines.close[1])
        # print("往后推2天（明后日）", 'datetime', self.data.lines.datetime.date(2), 'close', self.data.lines.close[2])
        print("已处理的数据点：", len(self.data))
        print("line的总长度：", self.data0.buflen())
        self.count += 1


class FuturesRollingStrategy(bt.Strategy):

    def __init__(self, reserve: float = 0.02, signal_df: Optional[pd.DataFrame] = None, signal_file: str = './input/signal/trade_AfterMovingContract.csv'):
        # 读取调仓表，表结构如下所示：
        # trade_date sec_code weight
        # 0 2019-01-31 000006.SZ 0.007282
        # 1 2019-01-31 000008.SZ 0.009783
        # ... ... ... ...
        # 2494 2021-01-28 688088.SH 0.007600
        self.reserve = reserve
        if signal_df is None:
            self.trade_assets = pd.read_csv(signal_file, parse_dates=['trade_date'])
            logger.info(f'`signal_df` is None, read signal from {signal_file}')
        else:
            self.trade_assets = signal_df
            logger.info(f'read signal from signal_df')
        # 读取调仓日期，即每月的最后一个交易日，回测时，会在这一天下单，然后在下一个交易日，以开盘价买入
        self.trade_dates = pd.to_datetime(self.trade_assets['trade_date'].unique()).tolist()
        self.order_list = []  # 记录以往订单，方便调仓日对未完成订单做处理
        self.trade_assets_pre = []  # 记录上一期持仓

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        dt = self.datas[0].datetime.date(0)  # 获取当前的回测时间点
        # 如果是调仓日，则进行调仓操作
        if dt in self.trade_dates:
            print("-------------- Trade Date: {}---------".format(dt))
            # 在调仓之前，取消之前所下的没成交也未到期的订单
            if len(self.order_list) > 0:
                for od in self.order_list:
                    self.cancel(od)  # 如果订单未完成，则撤销订单
                self.order_list = []  # 重置订单列表
            # 提取当前调仓日的持仓列表
            trade_assets_data = self.trade_assets.query(f"trade_date=='{dt}'")
            long_assets = trade_assets_data['sec_code'].tolist()
            print('long assets', long_assets)  # 打印持仓列表
            # 对现有持仓中，调仓后不再继续持有的股票进行卖出平仓
            sell_assets = [i for i in self.trade_assets_pre if i not in long_assets]
            print('sell_assets', sell_assets)  # 打印平仓列表
            if len(sell_assets) > 0:
                print("---------- Close Asset -------------")
                for asset in sell_assets:
                    data = self.getdatabyname(asset)
                    if self.getposition(data).size > 0:
                        od = self.close(data=data)
                        self.order_list.append(od)  # 记录卖出订单
            # 买入此次调仓的资产：多退少补原则
            print("---------- Trade Asset --------------")
            for asset in long_assets:
                w = trade_assets_data.query(f"sec_code=='{asset}'")['weight'].iloc[0]  # 提取持仓权重
                data = self.getdatabyname(asset)
                order = self.order_target_percent(data=data, target=w * self.reserve)  # 为减少可用资金不足的情况
                # order = self.order_target_size(data=data, target=20)
                self.order_list.append(order)

            self.trade_assets_pre = long_assets  # 保存此次调仓的股票列表
        # print(dt, f'{self.broker.getvalue():.2f}')
        # results = {data._name: self.getposition(data).size*self.getposition(data).adjbase for data in self.datas if self.getposition(data).size != 0}
        # print(results)

    def notify_trade(self, trade):
        if trade.isopen:
            self.log(f'Open Asset is {trade.getdataname()}, price : {trade.price}')
        if not trade.isclosed:
            return
        self.log(
            f'Close Asset is {trade.getdataname()}：\n Gross Return {trade.pnl:.2f}, Net Return {trade.pnlcomm:.2f}')

    #     @recordtordermsg
    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        elif order.status in [order.Completed]:
            # print(order.ref, num2date(order.created.dt), num2date(order.executed.dt), order.data._name, order.status)
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Asset: %s' %
                    (order.ref,  # 订单编号
                     order.executed.price,  # 成交价
                     order.executed.value,  # 成交额
                     order.executed.comm,  # 佣金
                     order.executed.size,  # 成交量
                     order.data._name))  # 股票名称
            else:  # Sell
                self.log('SELL EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Asset: %s' %
                         (order.ref,
                          order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          order.executed.size,
                          order.data._name))
        elif order.status in [order.Canceled]:
            self.log(f'CANCELED : order_ref: {order.ref}, data_name:{order.data._name}')
        elif order.status in [order.Rejected]:
            self.log(f'REJECT : order_ref: {order.ref}, data_name:{order.data._name}')
        elif order.status in [order.Margin]:
            self.log(f'MARGIN : order_ref: {order.ref}, data_name:{order.data._name}')
        elif order.status in [order.Partial]:
            self.log(f'PARTIAL : order_ref: {order.ref}, data_name:{order.data._name}')

    def stop(self):
        self.log(f'Final Value: {self.broker.getvalue():.2f}')
