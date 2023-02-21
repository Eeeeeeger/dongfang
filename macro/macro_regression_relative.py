import warnings

warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import *

'''
initialize data
'''
data = DataGenerator()
data.create_relative_labels()
features = data.features
assets = data.assets
assets_dict = data.assets_dict


def _regression(train_data: pd.DataFrame, predict_data: pd.DataFrame, indices: list,
                model: LogisticRegression):
    train_data = train_data.dropna()
    # standardize the features to converge fast
    ss = StandardScaler()
    train_data.loc[:, features] = ss.fit_transform(train_data.loc[:, features])
    predict_data.loc[:, features] = ss.transform(predict_data.loc[:, features])
    lr = model.fit(X=train_data.loc[:, features], y=train_data.loc[:, 0])

    # return with the dataframe with the probability of each class
    return pd.DataFrame(data=lr.predict_proba(predict_data),
                        index=predict_data.index,
                        columns=list(map(lambda x: assets_dict[x], lr.classes_))).reindex(indices).ffill()


'''
PARAMS FINE-TUNING PART IN TRAIN DATASET
'''

acc_lt, bt_lt = [], []

for label_day in [1, 3, 5, 20]:
    # choose the horizon by which label is calculated
    for model in ["LogisticRegression"]:
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
            for k, v in data.data_dict.items():
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
                    reg_data = pd.concat([train_data_dict[label_type], train_data_dict['input'][features]],
                                         axis=1).iloc[
                               predict_num - train_period:predict_num]
                    # predict dataset
                    predict_data = train_data_dict['input'][features].iloc[predict_num:predict_num + rolling_period]
                    indices = predict_data.index

                    # if the label horizon is bigger than one, than for each predicted value, it should last at least horizon days
                    predict_data = predict_data.iloc[::label_day]
                    # get prediction
                    temp = _regression(reg_data, predict_data, indices, m)

                    if ans.empty:
                        ans = temp
                    else:
                        ans = pd.concat([ans, temp], axis=0)

                logger.info('start evaluate on train set')
                ## 回测表现
                factor_ret = pd.Series(index=ans.index, name='ret', data=np.diagonal(
                    ans.idxmax(axis=1).apply(lambda x: data.data_dict['ret1d'].loc[:, x]).loc[:,
                    ans.index.tolist()])).dropna()
                summary = daily_ret_statistic(factor_ret)
                bt_lt.append(summary.iloc[-1])
                net_value_plot(ans, factor_ret, data.data_dict, assets, benchmark=True, save=True,
                               name=f"{label_day}_{model}_{train_period}_{rolling_period}",
                               path='./output/regression_relative/backtest/')
                ## 预测准确率
                # sign = train_data_dict['sign1d'].copy()
                # acc_rate = (ans.idxmax(axis=1).apply(lambda x:labels_dict[x]) == sign.loc[sign.index.intersection(ans.index)]).sum() / len(ans)
                # acc_lt.append(acc_rate)
bt_res = pd.concat(bt_lt, axis=1)
indices = pd.MultiIndex.from_product(
    [[1, 3, 5, 20], [120, 250, 500, 750], [1, 20, 60, 120]])
bt_res.columns = indices
# acc_res.loc['mean'] = acc_res.mean()
# acc_res.T.to_csv(f'output/regression_relative/acc.csv')
bt_res.T.to_csv(f'output/regression_relative/bt.csv')

'''
TEST PART
'''
# pick the top model according to sharpe ratio to get backtest results in test dataset
bt_res = pd.read_csv(f"output/regression_relative/bt.csv", index_col=[0, 1, 2]).T
for each in bt_res.T['sharpe'].nlargest(1).index.to_list():
    print(each)
    label_day, train_period, rolling_period = each
    if rolling_period < label_day:
        logger.debug('label_day is greater than rolling_period, reset rolling_period = label_day')
        rolling_period = label_day
    train_data_dict = {}
    test_data_dict = {}
    for k, v in data.data_dict.items():
        train_data_dict[k], test_data_dict[k] = v.iloc[:int(len(v) * 0.7)], v.iloc[
                                                                            int(len(v) * 0.7) - train_period:]

        label_type = f'sign{label_day}d'
        model = 'LogisticRegression'
        m = LogisticRegression(class_weight='balanced')

    logger.info(
        f'label_day: {label_day}, model: {model}, train_period: {train_period},rolling_period: {rolling_period}\n '
        f'start backtest on test set from {test_data_dict[label_type].index[train_period]} to {test_data_dict[label_type].index[-1]}')

    ans = pd.DataFrame()
    for predict_num in tqdm(range(train_period, len(test_data_dict['input']), rolling_period)):
        reg_data = pd.concat([test_data_dict[label_type], test_data_dict['input'][features]], axis=1).iloc[
                   predict_num - train_period:predict_num]
        predict_data = test_data_dict['input'][features].iloc[predict_num:predict_num + rolling_period]
        indices = predict_data.index
        predict_data = predict_data.iloc[::label_day]
        temp = _regression(reg_data, predict_data, indices, m)

        if ans.empty:
            ans = temp
        else:
            ans = pd.concat([ans, temp], axis=0)

    ## 回测表现
    logger.info('start evaluate on test set')
    factor_ret = pd.Series(index=ans.index, name='ret', data=np.diagonal(
        ans.idxmax(axis=1).apply(lambda x: data.data_dict['ret1d'].loc[:, x]).loc[:, ans.index.tolist()])).dropna()
    summary = daily_ret_statistic(factor_ret)
    summary.to_csv(f'output/regression_relative/test_{label_day}_{model}_{train_period}_{rolling_period}.csv')
    net_value_plot(ans, factor_ret, data.data_dict, assets, benchmark=True, save=True,
                   name=f"test_{label_day}_{model}_{train_period}_{rolling_period}",
                   path='./output/regression_relative/backtest/')
