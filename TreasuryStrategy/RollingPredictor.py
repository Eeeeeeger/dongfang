from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import *
from DataGenerator import DataGenerator
import lightgbm as lgb
import warnings
from collections import OrderedDict
from pathlib import Path
from loguru import logger

warnings.filterwarnings('ignore')


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    # 'objective': 'fair',
    'metric': 'l2',
    'max_depth': -1,
    'num_leaves': 32,
    'num_iterations': 200,
    'learning_rate': 0.001,
    'feature_fraction': 1,
    'bagging_fraction': 1,
    'bagging_freq': 5,
    'verbose': 1,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'device': 'cpu',
    'random_state': 100,
}

class RollingPredictor:
    def __init__(self,
                 data,
                 features,
                 label_day_lt=[1, 3, 5, 20],
                 train_period_lt=[120, 250, 500, 750],
                 rolling_period_lt=[1, 20, 60, 120],
                 ):
        self.data = data
        self.features = features
        self.label_day_lt, self.train_period_lt, self.rolling_period_lt = label_day_lt, train_period_lt, rolling_period_lt

    @staticmethod
    def _regression(train_data: pd.DataFrame,
                    predict_data: pd.DataFrame,
                    features: list,
                    target: str,
                    indices: list,
                    model):
        train_data = train_data.mask(np.isinf(train_data), float("nan"))
        train_data = train_data.dropna()
        # standardize the features to converge fast
        if model == 'LgbRegressor':
            x = np.array(train_data.loc[:, features])
            y = np.array(train_data.loc[:, target])
            valid_ratio = 0.3
            x_train = x[:int(len(x) * (1 - valid_ratio))]
            y_train = y[:int(len(x) * (1 - valid_ratio))]
            x_dev = x[int(len(x) * (1 - valid_ratio)):]
            y_dev = y[int(len(x) * (1 - valid_ratio)):]
            del train_data
            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_dev = lgb.Dataset(x_dev, y_dev, reference=lgb_train)
            del x_train, y_train, x_dev, y_dev
            tree = lgb.train(params, lgb_train, num_boost_round=1000,
                             valid_sets=lgb_dev, early_stopping_rounds=30)
            return pd.DataFrame(OrderedDict({target: tree.predict(predict_data)}), index=predict_data.index).reindex(
                indices).ffill()
        else:
            ss = StandardScaler()
            train_data.loc[:, features] = ss.fit_transform(train_data.loc[:, features])
            predict_data.loc[:, features] = ss.transform(predict_data.loc[:, features])
            lr = model.fit(X=train_data.loc[:, features], y=train_data.loc[:, target])

            # get prediction results for each asset
            if isinstance(model, LinearRegression):
                return pd.DataFrame(OrderedDict({target: lr.predict(predict_data.loc[:, features])}),
                                    index=predict_data.index).reindex(
                    indices).ffill()
            elif isinstance(model, LogisticRegression):
                return pd.DataFrame(OrderedDict({target: lr.predict_proba(predict_data.loc[:, features])[:, 1]}),
                                    index=predict_data.index).reindex(indices).ffill()


    def _fit_transform(self, data, label_type, train_period, rolling_period, label_day, m):
        ans = pd.DataFrame()
        for predict_num in tqdm(range(train_period, len(data), rolling_period)):
            # train times in train datasets
            # train dataset
            reg_data = pd.concat([data[label_type], data[self.features]],
                                 axis=1).iloc[
                       predict_num - train_period:predict_num]
            # predict dataset
            predict_data = data[self.features].iloc[predict_num:predict_num + rolling_period]
            indices = predict_data.index

            # if the label horizon is bigger than one, than for each predicted value, it should last at least horizon days
            predict_data = predict_data.iloc[::label_day]

            # use multiprocess to get prediction
            temp = self._regression(
                reg_data,
                predict_data,
                self.features,
                label_type,
                indices,
                m,
            )

            if ans.empty:
                ans = temp
            else:
                ans = pd.concat([ans, temp], axis=0)
        return ans

    def fitting(self, model):
        acc_lt, bt_lt = [], []
        for label_day in self.label_day_lt:
            # choose the horizon by which label is calculated
            if model == 'LinearRegression':
                label_type = f'ret{label_day}d'
                m = LinearRegression()
            elif model == 'LogisticRegression':
                label_type = f'sign{label_day}d'
                m = LogisticRegression(class_weight='balanced')
            elif model == 'LgbRegressor':
                label_type = f'sign{label_day}d'
                m = model
            else:
                raise ValueError

            for train_period in self.train_period_lt:
                # the length of train dataset
                # split train and test dataset

                train_data, test_data = self.data.iloc[
                                        :int(len(self.data) * 0.7)], self.data.iloc[
                                                                     int(len(
                                                                         self.data) * 0.7) - train_period:]

                for rolling_period in self.rolling_period_lt:
                    # the length of rolling period to retrain the model
                    if rolling_period < label_day:
                        logger.debug('label_day is greater than rolling_period, reset rolling_period = label_day')
                        rolling_period = label_day
                    # train_period = 250
                    # rolling_period = 100

                    logger.info(
                        f'label_day: {label_day}, model: {model}, train_period: {train_period},rolling_period: {rolling_period}\n'
                        f'start backtest on train set from {train_data.index[train_period]} to {train_data.index[-1]}')

                    ans = self._fit_transform(train_data, label_type, train_period, rolling_period, label_day, m)
                    logger.info('start evaluate on train set')
                    ## 回测表现
                    factor_ret = pd.Series(index=ans.index, name='ret',
                                           data=self.data['ret1d'].loc[ans.index]).dropna()
                    summary = daily_ret_statistic(factor_ret)
                    bt_lt.append(summary.iloc[-1])
                    net_value_plot(factor_ret, benchmark=True, save=True,
                                   name=f"{label_day}_{train_period}_{rolling_period}",
                                   path=f'./output/{model}/backtest/')

        self.bt_res = pd.concat(bt_lt, axis=1)
        indices = pd.MultiIndex.from_product(
            [[model], self.label_day_lt, self.train_period_lt, self.rolling_period_lt])
        self.bt_res.columns = indices
        self.bt_res.T.to_csv(f'output/{model}/bt.csv')

    def predict(self, top=1):
        for _params in self.bt_res.T['sharpe'].nlargest(top).index.to_list():
            model, label_day, train_period, rolling_period = _params
            if rolling_period < label_day:
                logger.debug('label_day is greater than rolling_period, reset rolling_period = label_day')
                rolling_period = label_day
            train_data, test_data = self.data.iloc[
                                    :int(len(self.data) * 0.7)], self.data.iloc[
                                                                 int(len(
                                                                     self.data) * 0.7) - train_period:]
            if model == 'LinearRegression':
                label_type = f'ret{label_day}d'
                m = LinearRegression()
            elif model == 'LogisticRegression':
                label_type = f'sign{label_day}d'
                m = LogisticRegression(class_weight='balanced')
            elif model == 'LgbRegressor':
                label_type = f'sign{label_day}d'
                m = model
            else:
                raise ValueError

            logger.info(
                f'label_day: {label_day}, model: {model}, train_period: {train_period},rolling_period: {rolling_period}\n '
                f'start backtest on test set from {train_data.index[train_period]} to {train_data.index[-1]}')

            ans = self._fit_transform(test_data, label_type, train_period, rolling_period, label_day, m)
            ## 回测表现
            logger.info('start evaluate on test set')
            factor_ret = pd.Series(index=ans.index, name='ret',
                                   data=self.data['ret1d'].loc[ans.index]).dropna()
            summary = daily_ret_statistic(factor_ret)
            summary.to_csv(f'output/{model}/test_{label_day}_{train_period}_{rolling_period}.csv')
            net_value_plot(factor_ret, benchmark=True, save=True,
                           name=f"test_{label_day}_{train_period}_{rolling_period}",
                           path=f'./output/{model}/backtest/')
