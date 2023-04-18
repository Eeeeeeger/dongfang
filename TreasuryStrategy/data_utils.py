'''
宏观数据处理
1. 缺失值线性填充
2. 季节性X-11
3. HP滤波/Kalman滤波
3. 同比差分匹配
4. Bootstrap 抽样回归/相关性检验
5. 降频/升频
6. 波动率倒数合成
'''
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import statsmodels.api as sm
from scipy.interpolate import interp1d
from statsmodels.tsa.filters.hp_filter import hpfilter
from collections import OrderedDict
from statsmodels.tsa.x13 import x13_arima_analysis
from dateutil.relativedelta import relativedelta
from pykalman import KalmanFilter

pd.set_option('display.max_columns', 100)
pd.set_option('expand_frame_repr', False)


def linear_interpolate(ts: pd.Series,
                       ) -> np.array or pd.Series:
    v = ts.copy()
    indices = v.index.tolist()
    v.reset_index(drop=True, inplace=True)
    x_new = v.index.tolist()
    x = v.dropna().index.tolist()
    y = v.dropna().values
    return pd.Series(index=indices,
                     data=interp1d(x, y, kind='linear', fill_value="extrapolate")(x_new)
                     )


def seasonality(ts: pd.Series,
                ) -> np.array:
    # ts, trend, abnormal = x13_arima_analysis(x, x12path='WinX13.exe')
    res = sm.tsa.seasonal_decompose(ts, model='additive', extrapolate_trend=True)
    return res.trend


def HP_filter(ts: np.array or pd.Series,
              lamb: float = 129600,
              ) -> tuple:
    cycle, trend = hpfilter(ts, lamb)
    return cycle


def Kalman1D(observations, ratio=1000):
    # To return the smoothed time series data
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    observation_covariance = transition_covariance*ratio
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state.reshape(-1), observations-pred_state.reshape(-1)


def match_log_dif(ts: pd.Series,
                  ) -> pd.Series:
    if (length := ts.groupby(ts.index.year).count().iloc[:-1].unique()).shape[0] == 1:
        return ts.pct_change(periods=length[0])*100
    else:
        res = ts.copy()
        for k, v in ts.groupby(ts.index.year).count().shift(1).to_dict().items():
            if np.isnan(v):
                res.loc[f'{k}'] = np.nan
            else:
                res.loc[f'{k}'] = ts.pct_change(int(v)).loc[f'{k}']*100
        return res


def preprocess(ts: pd.Series, interpolate: bool = False, dif: bool = False, season: bool = False):
    ts = ts.replace(0.0, np.nan)
    if interpolate and np.isnan(ts).any():
        ts = linear_interpolate(ts)
    else:
        ts = ts.dropna()
    if dif:
        ts = match_log_dif(ts).dropna()
    if season:
        ts = seasonality(ts)
    return ts


def linear_regression(df: pd.DataFrame,
                      y_col: str,
                      x_col: str or list,
                      intercept: bool = True,
                      ):
    data = df.copy()
    if isinstance(x_col, str):
        x_col = [x_col]
    data = data[[y_col]+x_col].dropna()
    y = data[y_col]
    x = data[x_col]
    if intercept:
        x = sm.add_constant(x)
    return sm.OLS(endog=y, exog=x).fit(cov_type='HAC', cov_kwds={'maxlags': 5})


def bootstrap_corr(data: pd.DataFrame,
                   y_col: str,
                   x_col: str or list,
                   nums: int = 1000,
                   ) -> pd.DataFrame:
    if isinstance(x_col, str):
        x_col = [x_col]
    new_data = data[x_col + [y_col]].dropna().copy()
    if (v := new_data.groupby(new_data.index.year).count().max().max()) == 1:
        rows_count = 2
    elif v == 4:
        rows_count = 8
    elif v == 12:
        rows_count = 24
    elif v > 200:
        rows_count = 500
    else:
        raise ValueError

    boot_res = []
    np.random.seed(100)
    for _ in range(nums):
        start_idx = np.random.choice(len(new_data) - rows_count)
        length_idx = rows_count + np.random.choice(len(new_data) - start_idx - rows_count)
        temp_data = new_data.iloc[start_idx:start_idx + length_idx].copy()
        boot_res.append(temp_data.corr().loc[y_col])
    ans = pd.concat(boot_res, axis=1).median(axis=1)
    ans = ans.loc[~(ans.index == y_col)]
    return ans


def bootstrap_regression(data: pd.DataFrame,
                         y_col: str,
                         x_col: str or list,
                         intercept: bool = True,
                         nums: int = 1000,
                         name: str = None,
                         ) -> pd.DataFrame:
    if isinstance(x_col, str):
        x_col = [x_col]
    new_data = data[x_col + [y_col]].dropna().copy()
    if (v := new_data.groupby(new_data.index.year).count().max().max()) == 1:
        rows_count = 2
    elif v == 4:
        rows_count = 8
    elif v == 12:
        rows_count = 24
    elif v > 200:
        rows_count = 500
    else:
        raise ValueError

    boot_res = []
    np.random.seed(100)
    if name is None:
        name = '+'.join(x_col)
    for _ in range(nums):
        start_idx = np.random.choice(len(new_data) - rows_count)
        length_idx = rows_count + np.random.choice(len(new_data) - start_idx - rows_count)
        temp_data = new_data.iloc[start_idx:start_idx + length_idx].copy()
        res = linear_regression(temp_data, y_col, x_col, intercept)
        boot_res.append(
            pd.concat([res.params.iloc[-1:], res.tvalues.iloc[-1:], pd.Series({name: res.rsquared_adj})], axis=1)
            )
    ans = pd.concat(boot_res, axis=0).median()
    ans.index = ['beta', 't', 'rsquared']
    ans.name = name
    return ans


def columns_rename(col):
    lt = col.split(':')
    return '_'.join([x for x in lt if '中国' not in x and '同比' not in x])


def select_by_regression(data: pd.DataFrame,
                         y: str,
                         x: list,
                         max_num: int,
                         ) -> list:
    steps = 1
    picked = [x[0]]
    unpicked = x[1:]
    print(steps, picked[0], 'is included...')
    base_rsq = bootstrap_regression(data, y, picked).loc['rsquared']
    while len(picked) < max_num:
        res = Parallel(n_jobs=len(unpicked))(
            delayed(bootstrap_regression)(
                data=data,
                y_col=y,
                x_col=picked + [_x],
                name=_x,
            )
            for _x in unpicked
        )
        res = pd.concat(res, axis=1).T

        if (v := res['rsquared'].max()) > base_rsq:
            steps = steps + 1
            k = res['rsquared'].idxmax()
            print(steps, k, 'is included...')
            picked.append(k)
            unpicked.remove(k)
            base_rsq = v
        else:
            break
    return picked


def synthesis_by_vol(data: pd.DataFrame,
                     rolling_period: int,
                     ) -> pd.Series:
    return data.mul(1/data.rolling(rolling_period).std().shift(1)).dropna(how='all').sum(axis=1)
