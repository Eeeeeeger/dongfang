from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Optional
from WindPy import *
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('bmh')
w.start()


def isTradingDayWind(day: str or pd.Timestamp) -> bool:
    if pd.Timestamp(day) in tradingDaysWind(day, day):
        return True
    return False


def isTradingDay(day: str or pd.Timestamp) -> bool:
    if pd.Timestamp(day) in pd.to_datetime(pd.read_csv('./input/trading_days.csv', index_col=0).index):
        return True
    return False


def prevTradingDayWind(dt: str or pd.Timestamp,
                       days: int = 1,
                       ) -> pd.Timestamp:
    if isTradingDayWind(dt):
        return pd.Timestamp(w.tdaysoffset(-days, dt, "").Data[0][0])
    else:
        return pd.Timestamp(w.tdaysoffset(-days + 1, dt, "").Data[0][0])


def prevTradingDay(dt: str or pd.Timestamp,
                   days: int = 1,
                   ) -> pd.Timestamp:
    if days > 40:
        raise ValueError('day is greater than 40 is not allowed')
    return tradingDays(start_date=pd.Timestamp(dt) - pd.Timedelta(days=60),
                end_date=pd.Timestamp(dt) - pd.Timedelta(days=1))[-days]


def nextTradingDayWind(dt: str or pd.Timestamp,
                       days: int = 1,
                       ) -> pd.Timestamp:
    return pd.Timestamp(w.tdaysoffset(days, dt, "").Data[0][0])


def nextTradingDay(dt: str or pd.Timestamp,
                   days: int = 1,
                   ) -> pd.Timestamp:
    if days > 40:
        raise ValueError('day is greater than 40 is not allowed')
    return tradingDays(start_date=pd.Timestamp(dt) + pd.Timedelta(days=1),
                       end_date=pd.Timestamp(dt) + pd.Timedelta(days=60))[days - 1]


def tradingDaysWind(start_date: str or pd.Timestamp,
                    end_date: str or pd.Timestamp,
                    ) -> List[pd.Timestamp]:
    if len(days := w.tdays(start_date, end_date, "").Data) == 0:
        return []
    else:
        return pd.to_datetime(days[0])


def tradingDays(start_date: str or pd.Timestamp or None = None,
                end_date: str or pd.Timestamp or None = None,
                ) -> List[pd.Timestamp]:
    trading_days = pd.to_datetime(pd.read_csv('./input/trading_days.csv', index_col=0).index)
    if start_date is None:
        start_date = trading_days[0]
    if end_date is None:
        end_date = trading_days[-1]

    if pd.Timestamp(start_date) < trading_days[0]:
        raise ValueError(f'start date is smaller than the {trading_days[0]}')
    elif pd.Timestamp(end_date) > trading_days[-1]:
        raise ValueError(f'end date is greater than {trading_days[-1]}')
    else:
        return trading_days[(trading_days >= start_date) & (trading_days <= end_date)]


def fetch_Wind_data(code: str or list,
                    col: str or list,
                    begin: str or pd.Timestamp,
                    end: str or pd.Timestamp
                    ) -> pd.DataFrame:
    if isinstance(begin, str):
        begin = pd.to_datetime(begin)
    if isinstance(end, str):
        end = pd.to_datetime(end)
    if isinstance(code, str):
        code = [code]
    if isinstance(col, str):
        col = [col]
    if len(code) == 1:
        ans = pd.DataFrame()
        begin_date = begin
        while (begin_date + timedelta(days=1500)) < end:
            end_date = (begin_date + timedelta(days=1500))
            data = w.wsd(code, col, begin_date, end_date)
            begin_date = (end_date + timedelta(days=1))
            temp_df = pd.DataFrame(data=data.Data, index=col,
                                   columns=pd.to_datetime(data.Times)).T.sort_index().dropna(how="all")
            ans = pd.concat([ans, temp_df], axis=0)
        data = w.wsd(code, col, begin_date, end)
        temp_df = pd.DataFrame(data=data.Data, index=col,
                               columns=pd.to_datetime(data.Times)).T.sort_index().dropna(how="all")
        ans = pd.concat([ans, temp_df], axis=0)
        return ans
    elif (isinstance(col, list) and len(col) == 1):
        ans = pd.DataFrame()
        begin_date = begin
        while (begin_date + timedelta(days=1500)) < end:
            end_date = (begin_date + timedelta(days=1500))
            data = w.wsd(code, col, begin_date, end_date)
            begin_date = (end_date + timedelta(days=1))
            temp_df = pd.DataFrame(data=data.Data, index=code,
                                   columns=pd.to_datetime(data.Times)).T.sort_index().dropna(how="all")
            ans = pd.concat([ans, temp_df], axis=0)

        data = w.wsd(code, col, begin_date, end)
        if begin_date == end:
            temp_df = pd.DataFrame(data=data.Data, index=pd.to_datetime(data.Times),
                                   columns=code).T.sort_index().dropna(how="all").T
        else:
            temp_df = pd.DataFrame(data=data.Data, index=code,
                                   columns=pd.to_datetime(data.Times)).T.sort_index().dropna(how="all")
        ans = pd.concat([ans, temp_df], axis=0)
        return ans
    else:
        raise ValueError('too much indicator to load')


def visualize_snowball(df: pd.DataFrame):
    fig = plt.figure(dpi=150)
    df['成立日'] = pd.to_datetime(df['成立日'])
    colors = {'已敲出未曾敲入': 'lightcoral', '已敲出曾敲入': 'orange', '已敲入且到期': 'lightblue',
              '未敲入且到期': 'darkcyan'}
    ax1 = fig.add_subplot(111)
    group = 0
    df['group'] = 0

    for ii in range(1, len(df)):
        if df.iloc[ii]['状态'] != df.iloc[ii - 1]['状态']:
            group += 1
        df.loc[df.index[ii], 'group'] = group

    cates = []
    for group in df['group'].unique():
        cate = df[df['group'] == group]['状态'].values[0]
        if cate not in cates:
            sns.lineplot(
                x='成立日',
                y='实际年化损益',
                data=df[df['group'] == group],
                color=colors[cate],
                label=cate,
                ax=ax1,
            )
            cates.append(cate)
        else:
            sns.lineplot(
                x='成立日',
                y='实际年化损益',
                data=df[df['group'] == group],
                color=colors[cate],
                ax=ax1,
            )
    mark = df.groupby('group')['成立日'].idxmin().values
    colors = np.random.rand(len(mark))
    # ax1.scatter(x=df.loc[mark, '成立日'], y=df.loc[mark, '实际年化损益'], c='navy', s=15, marker='o', alpha=0.6)
    ax1.set_ylabel('Snowball Annual Return')
    ax2 = ax1.twinx()
    zz500_df = fetch_Wind_data('000905.SH', 'close', df['成立日'].tolist()[0], df['成立日'].tolist()[-1])
    zz500_df.index.name = 'date'
    zz500_df.reset_index(inplace=True)
    sns.lineplot(
        x='date',
        y='close',
        data=zz500_df,
        ci=None,
        ax=ax2,
        color='grey',
    )
    ax2.set_ylabel('ZZ500')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(40)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(40)
    return fig


if __name__ == "__main__":
    # print(tradingDays('2022-01-10', '2022-01-15'))
    # print(tradingDays(pd.Timestamp('2022-01-10'), pd.Timestamp('2022-01-15')))
    # print(tradingDays(datetime.strptime('2022-01-10', "%Y-%m-%d"),
    #                   datetime.strptime('2022-01-15', "%Y-%m-%d")))
    # print(isTradingDay('2022-01-01'))
    # print(prevTradingDay('2022-01-01'))
    from constants import INDEX_CODE

    # print(fetch_Wind_data(INDEX_CODE['ZZ500'], ['close'], '2022-01-10', '2022-01-10')['close'].values[0])
    # visualize_snowball(pd.read_csv('./output/snowball2.csv')).savefig('./output/snowball.png')
    print((tradingDays('2020-01-01', '2022-01-01') == tradingDaysWind('20200101', '20220101')).all())
    for ii in range(1, 31):
        if prevTradingDayWind('20200101', ii) != prevTradingDay('20200101', ii):
            print(ii, 'prev')

        if nextTradingDay('20200101', ii) != nextTradingDayWind('20200101', ii):
            print(ii, 'next')
