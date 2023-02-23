import pandas as pd

from utils import *
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
import scipy.stats as stats
from loguru import logger

'''
initialize data
'''
data = DataGenerator()
data.create_labels()
features = data.features
assets = data.assets

train_data_dict = {}
test_data_dict = {}
for k, v in data.data_dict.items():
    train_data_dict[k], test_data_dict[k] = v.iloc[:int(len(v) * 0.7)], v.iloc[

                                                                        int(len(v) * 0.7):]
def adfuller_test(x):
    return True if adfuller(x)[1] < 0.05 else False


def granger_test(y, x):
    res = grangercausalitytests(
        x=pd.concat([y, x], axis=1), maxlag=5,
        verbose=False)
    res = [vv[1] for k, v in res.items() for kk, vv in v[0].items()]
    if (np.array(res) < 0.05).sum() >= 1:
        return True
    else:
        return False


def correlation_test(y, x):
    correlation, pvalue = stats.spearmanr(y, x)
    if pvalue < 0.05:
        return True
    else:
        return False


def trend_test(y, x):
    lt = []
    for para in [1,3,6]:
        trend = pd.Series(index=x.index, data=0)
        trend.loc[(np.sign(x.pct_change()).rolling(para).sum() == para)] = 1
        trend.loc[(np.sign(x.pct_change()).rolling(para).sum() == -para)] = -1
        trend.dropna(inplace=True)
        y_cor = y.loc[trend.index].copy()
        lt.append(correlation_test(y_cor, trend))
    return np.array(lt).any()


def rolling_test(y, x):
    lt = []
    for period in [20, 60, 120]:
        for para in [0.1, 0.2, 0.5]:
            rolling = pd.Series(index=x.index, data=0)
            rolling.loc[x.pct_change().rolling(period).sum() >= para] = 1
            rolling.loc[x.pct_change().rolling(period).sum() <= -para] = -1
            rolling.dropna(inplace=True)
            y_cor = y.loc[rolling.index].copy()
            lt.append(correlation_test(y_cor, rolling))
    return np.array(lt).sum() > int(len(lt)/2)


def extent_test(y, x):
    lt = []
    for para in [0.05, 0.1, 0.2]:
        extent = pd.Series(index=x.index, data=0)
        extent.loc[x.pct_change() >= para] = 1
        extent.loc[x.pct_change() <= -para] = -1
        extent.dropna(inplace=True)
        y_cor = y.loc[extent.index].copy()
        lt.append(correlation_test(y_cor, extent))
    return np.array(lt).any()


def zscore_test(y, x):
    lt = []
    for period in [60, 120]:
        def _zscore_percentile(x):
            temp = stats.zscore(x)
            if x.iloc[-1]<=temp.quantile(0.25):
                return -1
            elif x.iloc[-1]>=temp.quantile(0.75):
                return 1
            else:
                return 0
        zscore = x.rolling(period).apply(_zscore_percentile)
        zscore.dropna(inplace=True)
        y_cor = y.loc[zscore.index].copy()
        lt.append(correlation_test(y_cor, zscore))
    return np.array(lt).any()

ret = 'ret20d'

tests = pd.DataFrame(index=pd.MultiIndex.from_product([assets, features]),
                     columns=['adf', 'granger', 'spearman', 'trend', 'rolling', 'extent', 'zscore'],
                     data=0)

for feature in features:
    x = train_data_dict['input'][feature]
    if adfuller_test(x):
        tests.loc[[(asset, feature) for asset in assets], 'adf'] = 1
        for asset in assets:
            y = train_data_dict[ret][asset]
            if granger_test(y, x):
                tests.loc[(asset, feature), 'granger'] = 1
                # logger.info(f'{feature} pass granger test in predicting {asset}')
            else:
                tests.loc[(asset, feature), 'granger'] = 0
                # logger.debug(f'{feature} does not pass granger test in predicting {asset}')
                tests.loc[(asset, feature), 'spearman'] = 1 if correlation_test(y, x) else 0
                tests.loc[(asset, feature), 'trend'] = 1 if trend_test(y, x) else 0
                tests.loc[(asset, feature), 'rolling'] = 1 if rolling_test(y, x) else 0
                tests.loc[(asset, feature), 'extent'] = 1 if extent_test(y, x) else 0
                tests.loc[(asset, feature), 'zscore'] = 1 if zscore_test(y, x) else 0

    else:
        tests.loc[[(asset, feature) for asset in assets], 'adf'] = 0
        tests.loc[[(asset, feature) for asset in assets], 'granger'] = 0
        for asset in assets:
            y = train_data_dict[ret][asset]
            tests.loc[(asset, feature), 'spearman'] = 1 if correlation_test(y, x) else 0
            tests.loc[(asset, feature), 'trend'] = 1 if trend_test(y, x) else 0
            tests.loc[(asset, feature), 'rolling'] = 1 if rolling_test(y, x) else 0
            tests.loc[(asset, feature), 'extent'] = 1 if extent_test(y, x) else 0
            tests.loc[(asset, feature), 'zscore'] = 1 if zscore_test(y, x) else 0

print(tests)
