from base import snowBall
from utils import visualize_snowball, tradingDays
import pandas as pd
from loguru import logger
import os
logger.remove(handler_id=None)

# for ii, d in enumerate(tradingDays('20160101', '20220101')):
#     a = snowBall(f'产品{ii}', 'ZZ500', d, 12, 0.97, 0.1, 0.2)
#     a.update_daily()
#     a.save_results(f'./output/ZZ500_0.97.csv')
df = pd.read_csv('./output/ZZ500_0.97.csv')
print("ZZ500", len(df[df['状态'].str.startswith('已敲出')])/len(df))

for v in [-0.105]:
    # for ii, d in enumerate(tradingDays('20160101', '20220101')):
    #     a = snowBall(f'产品{ii}', '10Y', d, 12, v, 0.1, 0.2)
    #     a.update_daily()
    #     a.save_results(f'./output/10Y_{v}.csv')
    df = pd.read_csv(f'./output/10Y_{v}.csv')
    print("10Y", v, len(df[df['状态'].str.startswith('已敲出')])/len(df))
