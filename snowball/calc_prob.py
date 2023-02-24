import pandas as pd
import numpy as np
from dateutil.relativedelta import *
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')


def init_data(start_date='2013-01-31'):
    df = pd.read_csv('./input/10Y.csv', index_col=0, parse_dates=True).dropna()
    df = df.loc[start_date:]
    return df


def get_prob(df, method, bps, months):
    if method == 'end_out':
        pd.DataFrame(index=[method]).to_csv(Path() / file_name, mode='a')
        dates_list = df.index.to_list()
        new_bps = [-x for x in bps if x != 0] + bps
        new_bps.sort()
        ans = pd.DataFrame(index=new_bps,
                           columns=months)
        for period in months:
            total = 0
            cnt = {k: 0 for k in new_bps}
            for ii in range(len(df)):
                if dates_list[ii] > dates_list[-1] - relativedelta(months=period):
                    break
                temp = df.loc[dates_list[ii]:dates_list[ii] + relativedelta(months=period)]
                temp['amp'] = temp['close'] - temp['close'].iloc[0]

                for multi in new_bps:
                    amp = 0.01 * multi
                    if amp >= 0 and (temp['amp'].iloc[-1] > amp):
                        cnt[multi] = cnt[multi] + 1
                    elif amp < 0 and (temp['amp'].iloc[-1] < amp):
                        cnt[multi] = cnt[multi] + 1
                total = total + 1
            for multi in new_bps:
                ans.loc[multi, period] = cnt[multi] / total
        ans.to_csv(Path() / file_name, mode='a')
    elif method == 'not_out':
        pd.DataFrame(index=[method]).to_csv(Path() / file_name, mode='a')
        dates_list = df.index.to_list()
        ans = pd.DataFrame(index=bps,
                           columns=months)
        for period in months:
            total = 0
            cnt = {k: 0 for k in bps}
            for ii in range(len(df)):
                if dates_list[ii] > dates_list[-1] - relativedelta(months=period):
                    break
                temp = df.loc[dates_list[ii]:dates_list[ii] + relativedelta(months=period)]
                temp['amp'] = temp['close'] - temp['close'].iloc[0]

                for multi in bps:
                    amp = 0.01 * multi
                    if (temp['amp'] >= -amp).all() and (temp['amp'] <= amp).all():
                        cnt[multi] = cnt[multi] + 1

                total = total + 1
            for multi in bps:
                ans.loc[multi, period] = cnt[multi] / total
        ans.to_csv(Path() / file_name, mode='a')


file_name = 'bp_prob.csv'
method = ['end_out', 'not_out'][0]
bps = [30, 20, 15, 10, 5, 0]
months = [1, 2, 3, 6, 12]
if (Path() / file_name).exists():
    os.remove((Path() / file_name))

df = init_data()
get_prob(df, 'end_out', bps, months)
get_prob(df, 'not_out', bps, months)
