import pandas as pd
import numpy as np
from itertools import combinations


def ma(data, cols, days):
    if cols is None:
        df = data.copy()
    else:
        df = data[cols].copy()
    df = df.rolling(days).mean()
    df.columns = [x + f'_ma{days}' for x in cols]
    return df


def std(data, cols, days):
    df = data[cols].copy()
    df = df.rolling(days).std()
    df.columns = [x + f'_std{days}' for x in data.columns]
    return df


def diff_ts(data, cols, name):
    df = pd.DataFrame(index=data.index)
    for _comb in combinations(cols, 2):
        df[name + '_' + _comb[0].split(':')[-1] + _comb[1].split(':')[-1]] = data[_comb[0]] - data[_comb[1]]
    return df


def diff_cs(data, col1, col2, name):
    df = pd.DataFrame(index=data.index)
    if len(col1) != len(col2):
        raise ValueError
    for ii in range(len(col1)):
        df[name + '_' + col1[ii].split(':')[-1] + col2[ii].split(':')[-1]] = data[col1[ii]] - data[col2[ii]]
    return df


def fd(data, cols):
    if cols is None:
        df = data.copy()
    else:
        df = data[cols].copy()
    df = df.diff()
    df.columns = [x + f'_fd' for x in cols]


class DataGenerator:
    def __init__(self, path='./MacroData.xlsx'):
        self.df = pd.read_excel(path, sheet_name='Money', skiprows=1, index_col=0, parse_dates=True)
        self.df.columns = [x.replace("中国:", "") for x in self.df.columns]
        self.features_treasury = ['中债国债到期收益率:1年', '中债国债到期收益率:2年', '中债国债到期收益率:3年',
                                  '中债国债到期收益率:4年',
                                  '中债国债到期收益率:5年', '中债国债到期收益率:6年', '中债国债到期收益率:7年',
                                  '中债国债到期收益率:8年',
                                  '中债国债到期收益率:10年', '中债国债到期收益率:20年']
        self.features_treasury_bond = ['中债国开债到期收益率:1年', '中债国开债到期收益率:2年',
                                       '中债国开债到期收益率:3年',
                                       '中债国开债到期收益率:4年', '中债国开债到期收益率:5年',
                                       '中债国开债到期收益率:6年',
                                       '中债国开债到期收益率:7年',
                                       '中债国开债到期收益率:8年', '中债国开债到期收益率:10年',
                                       '中债国开债到期收益率:20年']
        self.features_corporate_bond = ['中债企业债到期收益率(AAA):1年',
                                        '中债企业债到期收益率(AAA):2年', '中债企业债到期收益率(AAA):3年',
                                        '中债企业债到期收益率(AAA):4年',
                                        '中债企业债到期收益率(AAA):5年', '中债企业债到期收益率(AAA):6年',
                                        '中债企业债到期收益率(AAA):7年',
                                        '中债企业债到期收益率(AAA):8年', '中债企业债到期收益率(AAA):10年',
                                        '中债企业债到期收益率(AAA):20年']
        self.features_shibor = ['SHIBOR:隔夜',
                                'SHIBOR:1个月', 'SHIBOR:3个月', 'SHIBOR:6个月', 'SHIBOR:9个月', 'SHIBOR:1年']
        self.features_r00 = ['R001', 'R007']
        self.features_usa = ['美元指数']

        self.macro_df = pd.concat([pd.read_excel(path, sheet_name='Macro_week', skiprows=1, index_col=0, parse_dates=True),
                                   pd.read_excel(path, sheet_name='Macro_day', skiprows=1, index_col=0, parse_dates=True),
                                   pd.read_excel(path, sheet_name='Macro_month', skiprows=1, index_col=0, parse_dates=True),
                                   ], axis=1).replace(0.0, float("nan")).ffill()
        self.df = self.df.join(self.macro_df, how='outer').loc[self.df['中债国债到期收益率:10年'].replace(0.0, np.nan).dropna().index]
        self.df.drop(['(停止)中国铁矿石价格指数(CIOPI)', '农产品批发价格200指数', '中国:水泥发运率:全国:当周值'], axis=1, inplace=True)
        for each in self.df.columns:
            print(each, self.df[each].dropna().index[0])
        self.features = self.df.columns.to_list()

    def calc_derivative_factors(self):
        self.derivative_df = pd.concat([self.df,
                                        fd(self.df, self.df.columns.to_list()),
                                        ma(self.df, self.df.columns.to_list(), 5),
                                        ma(self.df, self.df.columns.to_list(), 10),
                                        ma(self.df, self.df.columns.to_list(), 20),
                                        std(self.df, self.df.columns.to_list(), 5),
                                        std(self.df, self.df.columns.to_list(), 10),
                                        std(self.df, self.df.columns.to_list(), 20),
                                        diff_ts(self.df, self.features_treasury, '中债国债到期收益率期限结构'),
                                        diff_ts(self.df, self.features_treasury_bond, '中债国开债到期收益率期限结构'),
                                        diff_ts(self.df, self.features_corporate_bond,
                                                '中债企业债(AAA)到期收益率期限结构'),
                                        diff_cs(self.df, self.features_treasury, self.features_treasury_bond,
                                                '税收利差')
                                        ], axis=1)
        self.features = self.derivative_df.columns.to_list()

    def calc_label(self, target='中债国债到期收益率:10年', derivative=True,  start_date='2015-01-01', end_date='2023-03-01'):
        self.target = target
        if derivative:
            self.derivative_df['ret1d'] = self.derivative_df[target].pct_change(1).shift(-1)
            self.derivative_df['ret3d'] = self.derivative_df[target].pct_change(3).shift(-3)
            self.derivative_df['ret5d'] = self.derivative_df[target].pct_change(5).shift(-5)
            self.derivative_df['ret20d'] = self.derivative_df[target].pct_change(20).shift(-20)
            self.derivative_df['sign1d'] = np.sign(self.derivative_df['ret1d'])
            self.derivative_df['sign3d'] = np.sign(self.derivative_df['ret3d'])
            self.derivative_df['sign5d'] = np.sign(self.derivative_df['ret5d'])
            self.derivative_df['sign20d'] = np.sign(self.derivative_df['ret20d'])
            self.derivative_df = self.derivative_df.loc[start_date:end_date]

        else:
            self.df['ret1d'] = self.df[target].pct_change(1).shift(-1)
            self.df['ret3d'] = self.df[target].pct_change(3).shift(-3)
            self.df['ret5d'] = self.df[target].pct_change(5).shift(-5)
            self.df['ret20d'] = self.df[target].pct_change(20).shift(-20)
            self.df['sign1d'] = np.sign(self.df['ret1d'])
            self.df['sign3d'] = np.sign(self.df['ret3d'])
            self.df['sign5d'] = np.sign(self.df['ret5d'])
            self.df['sign20d'] = np.sign(self.df['ret20d'])
            self.df = self.df.loc[start_date:end_date]
