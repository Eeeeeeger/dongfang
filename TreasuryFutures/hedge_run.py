from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from DynamicHedge.StatisticsModel import OLSModel, WLSModel, VARModel

#
# freq_lt = ['daily','monthly']
# trace_lt = [10, 20, 40, 60, 80, 100]
# train, valid, test = [],[],[]
# hedge_cls = OLSModel(trade_type=True)
# for freq in freq_lt:
#     for trace in trace_lt:
#         res = hedge_cls.hedge(frequency=freq, trace_back=trace, **{'const': True}).set_index('index')
#         valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
#         test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())
#
# hedge_cls = WLSModel(trade_type=True)
# for freq in freq_lt:
#     for trace in trace_lt:
#         res = hedge_cls.hedge(frequency=freq, trace_back=trace, halflife=int(trace/2), **{'const': True}).set_index('index')
#         valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
#         test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())
#
# hedge_cls = WLSModel(trade_type=True)
# for freq in freq_lt:
#     for trace in trace_lt:
#         res = hedge_cls.hedge(frequency=freq, trace_back=trace, halflife=int(trace/4), **{'const': True}).set_index('index')
#         valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
#         test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())
#
# hedge_cls = VARModel(trade_type=True)
# for freq in freq_lt:
#     for trace in trace_lt:
#         res = hedge_cls.hedge(frequency=freq, trace_back=trace, **{'const': True}).set_index('index')
#         valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
#         test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())
# df = pd.DataFrame({'valid': valid, 'test': test},
#                   index=pd.MultiIndex.from_product([['ols', 'wls_2', 'wls_4', 'var'], freq_lt, trace_lt]))
# df.to_csv('./output/hedge/model_val.csv')

# trend
leverage_lt = [1.2, 1.5]
threshold_lt = [0.6, 0.65, 0.7]
adj_days_lt =[5, 10]
freq = 'daily'
trace = 80
valid, test = [], []
hedge_cls = OLSModel(trade_type=True)
for leverage in leverage_lt:
    for threshold in threshold_lt:
        for adj_days in adj_days_lt:
            res = hedge_cls.hedge(frequency=freq, trace_back=trace,
                                  **{'const': True, 'adj_method': 'trend', 'adj_days': adj_days,'leverage': leverage, 'threshold': threshold}).set_index('index')
            valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
            test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())

hedge_cls = WLSModel(trade_type=True)
for leverage in leverage_lt:
    for threshold in threshold_lt:
        for adj_days in adj_days_lt:
            res = hedge_cls.hedge(frequency=freq, trace_back=trace, halflife=int(trace/2),
                                  **{'const': True, 'adj_method': 'trend', 'adj_days': adj_days,'leverage': leverage, 'threshold': threshold}).set_index('index')
            valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
            test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())

hedge_cls = WLSModel(trade_type=True)
for leverage in leverage_lt:
    for threshold in threshold_lt:
        for adj_days in adj_days_lt:
            res = hedge_cls.hedge(frequency=freq, trace_back=trace, halflife=int(trace/4),
                                  **{'const': True, 'adj_method': 'trend', 'adj_days': adj_days,'leverage': leverage, 'threshold': threshold}).set_index('index')
            valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
            test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())

hedge_cls = VARModel(trade_type=True)
for leverage in leverage_lt:
    for threshold in threshold_lt:
        for adj_days in adj_days_lt:
            res = hedge_cls.hedge(frequency=freq, trace_back=trace,
                                  **{'const': True, 'adj_method': 'trend', 'adj_days': adj_days,
                                     'leverage': leverage, 'threshold': threshold}).set_index('index')
            valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
            test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())

df = pd.DataFrame({'valid': valid, 'test': test},
                  index=pd.MultiIndex.from_product([['ols', 'wls_2', 'wls_4', 'var'], leverage_lt, threshold_lt, adj_days_lt]))
df.to_csv('./output/hedge/trend_adj.csv')


# std
leverage_lt = [1.2, 1.5]
threshold_lt = [0.15, 0.18, 0.2]
adj_days_lt =[5, 10]
freq = 'daily'
trace = 80
valid, test = [], []
hedge_cls = OLSModel(trade_type=True)
for leverage in leverage_lt:
    for threshold in threshold_lt:
        for adj_days in adj_days_lt:
            res = hedge_cls.hedge(frequency=freq, trace_back=trace,
                                  **{'const': True, 'adj_method': 'std', 'adj_days': adj_days,'leverage': leverage, 'threshold': threshold}).set_index('index')
            valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
            test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())

hedge_cls = WLSModel(trade_type=True)
for leverage in leverage_lt:
    for threshold in threshold_lt:
        for adj_days in adj_days_lt:
            res = hedge_cls.hedge(frequency=freq, trace_back=trace, halflife=int(trace/2),
                                  **{'const': True, 'adj_method': 'std', 'adj_days': adj_days,'leverage': leverage, 'threshold': threshold}).set_index('index')
            valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
            test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())

hedge_cls = WLSModel(trade_type=True)
for leverage in leverage_lt:
    for threshold in threshold_lt:
        for adj_days in adj_days_lt:
            res = hedge_cls.hedge(frequency=freq, trace_back=trace, halflife=int(trace/4),
                                  **{'const': True, 'adj_method': 'std', 'adj_days': adj_days,'leverage': leverage, 'threshold': threshold}).set_index('index')
            valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
            test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())

hedge_cls = VARModel(trade_type=True)
for leverage in leverage_lt:
    for threshold in threshold_lt:
        for adj_days in adj_days_lt:
            res = hedge_cls.hedge(frequency=freq, trace_back=trace,
                                  **{'const': True, 'adj_method': 'std', 'adj_days': adj_days,
                                     'leverage': leverage, 'threshold': threshold}).set_index('index')
            valid.append(res['variance reduction'].iloc[:5].astype(np.float64).mean())
            test.append(res['variance reduction'].iloc[5:8].astype(np.float64).mean())

df = pd.DataFrame({'valid': valid, 'test': test},
                  index=pd.MultiIndex.from_product([['ols', 'wls_2', 'wls_4', 'var'], leverage_lt, threshold_lt, adj_days_lt]))
df.to_csv('./output/hedge/std_adj.csv')