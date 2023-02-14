'''
use pct
'''
import warnings

warnings.filterwarnings('ignore')
# 绘制图形
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import *

'''
initialize data
'''
factors = pd.read_csv('./input/macroFactorDf20230202_1.csv', index_col=0, parse_dates=True)
targets = pd.read_csv('./input/index.csv', index_col=0, parse_dates=True)
data = pd.concat([factors, targets], axis=1)
data = pd.concat([data.iloc[:-1].dropna(), data.iloc[-1:]], axis=0)
assets = targets.columns.tolist()
assets = ['000905.SH', 'NH0100.NHF', 'Bond']
labels_dict = dict(zip(assets, [-1, 0, 1]))
macro = factors.columns.tolist()

# merge new features by first differencing
data = data.join(data[macro].diff().rename(columns=dict(zip(macro, map(lambda x: f'd_{x}', macro)))))
features = macro + list(map(lambda x: f'd_{x}', macro))
data_dict = {'input': data.copy(), 'ret1d': data[assets].copy(), 'ret3d': data[assets].copy(),
             'ret5d': data[assets].copy(),
             'ret20d': data[assets].copy(),
             'sign1d': data[assets].copy(), 'sign3d': data[assets].copy(), 'sign5d': data[assets].copy(),
             'sign20d': data[assets].copy()}

data.dropna(subset=features, how='any', inplace=True)
dates = data.index.tolist()

# generate label tag
for asset in assets:
    data_dict['ret1d'][asset] = data_dict['input'][asset].pct_change(1).shift(-1)
    data_dict['ret3d'][asset] = data_dict['input'][asset].pct_change(3).shift(-3)
    data_dict['ret5d'][asset] = data_dict['input'][asset].pct_change(5).shift(-5)
    data_dict['ret20d'][asset] = data_dict['input'][asset].pct_change(20).shift(-20)
    data_dict['sign1d'][asset] = np.sign(data_dict['ret1d'][asset])
    data_dict['sign3d'][asset] = np.sign(data_dict['ret3d'][asset])
    data_dict['sign5d'][asset] = np.sign(data_dict['ret5d'][asset])
    data_dict['sign20d'][asset] = np.sign(data_dict['ret20d'][asset])

# padding the index
for k, v in data_dict.items():
    data_dict[k] = v.loc[v.index.intersection(data.index)]
del data


def _regression(train_data: pd.DataFrame, predict_data: pd.DataFrame, asset: str, indices: list,
                model: LinearRegression or LogisticRegression):
    train_data = train_data[[asset] + features].dropna()
    # standardize the features to converge fast
    ss = StandardScaler()
    train_data.loc[:, features] = ss.fit_transform(train_data.loc[:, features])
    predict_data.loc[:, features] = ss.transform(predict_data.loc[:, features])
    lr = model.fit(X=train_data.loc[:, features], y=train_data.loc[:, asset])

    # get prediction results for each asset
    if isinstance(model, LinearRegression):
        return pd.DataFrame(OrderedDict({asset: lr.predict(predict_data)}), index=predict_data.index).reindex(
            indices).ffill()
    elif isinstance(model, LogisticRegression):
        return pd.DataFrame(OrderedDict({asset: lr.predict_proba(predict_data)[:, 1]}),
                            index=predict_data.index).reindex(indices).ffill()


'''
PARAMS FINE-TUNING PART IN TRAIN DATASET
'''

acc_lt, bt_lt = [], []

for label_day in [1, 3, 5, 20]:
    # choose the horizon by which label is calculated
    for model in ["LinearRegression", "LogisticRegression"]:
        # choose the used model
        if model == 'LinearRegression':
            label_type = f'ret{label_day}d'
            m = LinearRegression()
        elif model == 'LogisticRegression':
            label_type = f'sign{label_day}d'
            m = LogisticRegression(class_weight='balanced')
        else:
            raise ValueError

        for train_period in [120, 250, 500, 750]:
            # the length of train dataset
            # split train and test dataset
            train_data_dict = {}
            test_data_dict = {}
            for k, v in data_dict.items():
                train_data_dict[k], test_data_dict[k] = v.iloc[:int(len(v) * 0.7)], v.iloc[
                                                                                    int(len(v) * 0.7) - train_period:]

            for rolling_period in [1, 20, 60, 120]:
                # the length of rolling period to retrain the model
                if rolling_period < label_day:
                    logger.debug('label_day is greater than rolling_period, reset rolling_period = label_day')
                    rolling_period = label_day
                # train_period = 250
                # rolling_period = 100
                ans = pd.DataFrame()
                logger.info(
                    f'label_day: {label_day}, model: {model}, train_period: {train_period},rolling_period: {rolling_period}\n'
                    f'start backtest on train set from {train_data_dict[label_type].index[train_period]} to {train_data_dict[label_type].index[-1]}')

                for predict_num in tqdm(range(train_period, len(train_data_dict['input']), rolling_period)):
                    # train times in train datasets
                    # train dataset
                    reg_data = pd.concat([train_data_dict[label_type][assets], train_data_dict['input'][features]],
                                         axis=1).iloc[
                               predict_num - train_period:predict_num]
                    # predict dataset
                    predict_data = train_data_dict['input'][features].iloc[predict_num:predict_num + rolling_period]
                    indices = predict_data.index

                    # if the label horizon is bigger than one, than for each predicted value, it should last at least horizon days
                    predict_data = predict_data.iloc[::label_day]

                    # use multiprocess to get prediction
                    temp = Parallel(n_jobs=len(assets))(
                        delayed(_regression)(
                            reg_data,
                            predict_data,
                            asset,
                            indices,
                            m,
                        )
                        for asset in assets
                    )

                    temp = pd.concat(temp, axis=1)
                    if ans.empty:
                        ans = temp
                    else:
                        ans = pd.concat([ans, temp], axis=0)

                logger.info('start evaluate on train set')
                ## 回测表现
                factor_ret = pd.Series(index=ans.index, name='ret', data=np.diagonal(
                    ans.idxmax(axis=1).apply(lambda x: data_dict['ret1d'].loc[:, x]).loc[:,
                    ans.index.tolist()])).dropna()
                summary = daily_ret_statistic(factor_ret)
                bt_lt.append(summary.iloc[-1])
                net_value_plot(ans, factor_ret, data_dict, assets, benchmark=True, save=True,
                               name=f"{label_day}_{model}_{train_period}_{rolling_period}",
                               path='./output/regression/backtest/')
                ## 预测准确率
                sign = train_data_dict['sign1d'].copy().loc[:, assets]
                acc_rate = (np.sign(ans) == np.sign(sign).loc[sign.index.intersection(ans.index)]).sum() / len(ans)
                acc_lt.append(acc_rate)

acc_res, bt_res = pd.concat(acc_lt, axis=1), pd.concat(bt_lt, axis=1)
indices = pd.MultiIndex.from_product(
    [[1, 3, 5, 20], ['LinearRegression', 'LogisticRegression'], [120, 250, 500, 750], [1, 20, 60, 120]])
acc_res.columns, bt_res.columns = indices, indices
acc_res.loc['mean'] = acc_res.mean()
acc_res.T.to_csv(f'output/regression/acc.csv')
bt_res.T.to_csv(f'output/regression/bt.csv')

'''
TEST PART
'''
# pick the top model according to sharpe ratio to get backtest results in test dataset
bt_res = pd.read_csv(f"output/regression/bt.csv", index_col=[0, 1, 2, 3]).T
for each in bt_res.T['sharpe'].nlargest(1).index.to_list():
    print(each)
    label_day, model, train_period, rolling_period = each
    if rolling_period < label_day:
        logger.debug('label_day is greater than rolling_period, reset rolling_period = label_day')
        rolling_period = label_day
    train_data_dict = {}
    test_data_dict = {}
    for k, v in data_dict.items():
        train_data_dict[k], test_data_dict[k] = v.iloc[:int(len(v) * 0.7)], v.iloc[
                                                                            int(len(v) * 0.7) - train_period:]

    if model == 'LinearRegression':
        label_type = f'ret{label_day}d'
        m = LinearRegression()
    elif model == 'LogisticRegression':
        label_type = f'sign{label_day}d'
        m = LogisticRegression(class_weight='balanced')
    else:
        raise ValueError

    logger.info(
        f'label_day: {label_day}, model: {model}, train_period: {train_period},rolling_period: {rolling_period}\n '
        f'start backtest on test set from {test_data_dict[label_type].index[train_period]} to {test_data_dict[label_type].index[-1]}')

    ans = pd.DataFrame()
    for predict_num in tqdm(range(train_period, len(test_data_dict['input']), rolling_period)):
        reg_data = pd.concat([test_data_dict[label_type][assets], test_data_dict['input'][features]], axis=1).iloc[
                   predict_num - train_period:predict_num]
        predict_data = test_data_dict['input'][features].iloc[predict_num:predict_num + rolling_period]
        indices = predict_data.index
        predict_data = predict_data.iloc[::label_day]
        temp = Parallel(n_jobs=len(assets))(
            delayed(_regression)(
                reg_data,
                predict_data,
                asset,
                indices,
                m,
            )
            for asset in assets
        )

        temp = pd.concat(temp, axis=1)
        if ans.empty:
            ans = temp
        else:
            ans = pd.concat([ans, temp], axis=0)

    ## 回测表现
    logger.info('start evaluate on test set')
    factor_ret = pd.Series(index=ans.index, name='ret', data=np.diagonal(
        ans.idxmax(axis=1).apply(lambda x: data_dict['ret1d'].loc[:, x]).loc[:, ans.index.tolist()])).dropna()
    summary = daily_ret_statistic(factor_ret)
    summary.to_csv(f'output/regression/test_{label_day}_{model}_{train_period}_{rolling_period}.csv')
    net_value_plot(ans, factor_ret, data_dict, assets, benchmark=True, save=True,
                   name=f"test_{label_day}_{model}_{train_period}_{rolling_period}",
                   path='./output/regression/backtest/')
