import pandas as pd
import numpy as np
from WindPy import *
w.start()


start_date = '2016-01-01'
end_date = '2023-02-01'
data = {}
for code in ['T00.CFE', 'T01.CFE', 'T02.CFE']:
    data[code] = w.wsd(code, "pre_close,close,pre_settle,settle,volume,amt,oi,trade_hiscode", start_date, end_date,"",usedf=True)[1].assign(code=code)

# full_data = pd.concat(list(data.values()), axis=0)
# contracts_list = full_data['TRADE_HISCODE'].unique()
# contracts_list.sort()
# contract_date_dict = {contract: (full_data.query(f'TRADE_HISCODE == "{contract}"').index.min(), full_data.query(f'TRADE_HISCODE == "{contract}"').index.max()) for contract in contracts_list}
# dominant_date_dict = {contract: (data['T00.CFE'].query(f'TRADE_HISCODE == "{contract}"').index.min(), data['T00.CFE'].query(f'TRADE_HISCODE == "{contract}"').index.max()) for contract in data['T00.CFE']['TRADE_HISCODE'].unique()}
# second_date_dict = {contract: (data['T01.CFE'].query(f'TRADE_HISCODE == "{contract}"').index.min(), data['T01.CFE'].query(f'TRADE_HISCODE == "{contract}"').index.max()) for contract in data['T01.CFE']['TRADE_HISCODE'].unique()}
# third_date_dict = {contract: (data['T02.CFE'].query(f'TRADE_HISCODE == "{contract}"').index.min(), data['T02.CFE'].query(f'TRADE_HISCODE == "{contract}"').index.max()) for contract in data['T02.CFE']['TRADE_HISCODE'].unique()}


def query_ctd(contract, start_date=None, end_date=None):
    """ctd，根据中债估值计算"""
    df = w.wsd(contract, "tbf_CTD2", start_date, end_date, "exchangeType=NIB;bondPriceType=1", usedf=True)[1]
    df.columns = ['ctd']
    """cf"""
    bond_codes = list(df.iloc[:, 0].unique())
    bond_codes = [str(i) for i in bond_codes]   # update当日，CTD结果未出，codes含None，query cf报错
    df_cf = w.wsd(','.join(bond_codes), "tbf_cvf", start_date, start_date, f"contractCode={contract}", usedf=True)[1]
    df_cf.columns = ['convention_factor']
    df = df.merge(df_cf, left_on='ctd', right_index=True, how='left')
    return df


def query_bond(self, bond_id, start_date=None, end_date=None):
    """
    读取df行情
    w.wsd("T2212.CFE", "tbf_CTD2", "2022-09-24", "2022-10-23", "exchangeType=NIB;bondPriceType=1")
    :param start_date:
    :param end_date:
    :return:
    """
    from WindPy import w
    w.start()
    """bond_full_px, duration"""
    df_bond = w.wsd(bond_id, ','.join([k for k, v in self.bond_item_dict.items()]),
                    start_date, end_date, "returnType=1;credibility=1", usedf=True)[1]
    df_bond.rename(self.bond_item_dict, axis=1, inplace=True)
    return df_bond.dropna()