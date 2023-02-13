from .BaseModel import *
from WindPy import *
w.start()
'''
1. 利用CTD券替代国债期货
2. 利用`基点价值/DV01` or `修正久期` 计算套保比率
3. 收益率beta的动态调整
国信证券-国债期货套保策略研究
'''


class DurationModel(BaseModel):
    def __init__(self,
                 dominant_type: str = 'AfterMovingContract',
                 days: int = 3,
                 frequency: str or int = 'daily',
                 ):
        super(DurationModel, self).__init__(dominant_type, days)
        self.frequency = frequency
        self.contract_period = self.futures.reset_index().groupby('code')['date'].agg(['first', 'last']).reset_index()
        self.ctd = pd.concat([self.query(contract=self.contract_period.iloc[ii]['code'],
                                         start_date=self.contract_period.iloc[ii]['first'],
                                         end_date=self.contract_period.iloc[ii]['last']
                                         ) for ii in range(len(self.contract_period))], axis=0)
    # 寻找CTD券
    @staticmethod
    def query(contract: str,
              start_date: Optional[str] = None,
              end_date: Optional[str] = None
              ) -> pd.DataFrame:
        """
        读取df行情
        w.wsd("T2212.CFE", "tbf_CTD2", "2022-09-24", "2022-10-23", "exchangeType=NIB;bondPriceType=1")
        :return:
                    CTD         convention_factor ...
        2021-09-24  200006.IB   0.9763
        ...         ...         ...
        2021-10-11  2000004.IB  0.9895
        """
        w.start()
        """ctd，根据中债估值计算"""
        df = w.wsd(contract, "tbf_CTD2", start_date, end_date, "exchangeType=NIB;bondPriceType=1", usedf=True)[1]
        df.columns = ['ctd']
        """cf"""
        bond_codes = list(df.iloc[:, 0].unique())
        bond_codes = [str(i) for i in bond_codes]  # update当日，CTD结果未出，codes含None，query cf报错
        df_cf = w.wsd(','.join(bond_codes), "tbf_cvf", start_date, start_date, f"contractCode={contract}", usedf=True)[
            1]
        df = df.merge(df_cf, left_on='ctd', right_index=True, how='left').set_index('ctd', append=True)
        df_info = []
        for bond in bond_codes:
            df_info.append(
                w.wsd(bond, "close,modifiedduration,net_cnbd,dirty_cnbd,yield_cnbd,modidura_cnbd,vobp_cnbd", start_date,
                      end_date, "credibility=1", usedf=True)[1]
                .assign(ctd=bond)
                .set_index('ctd', append=True)
            )
        df_info = pd.concat(df_info, axis=0)
        df = df.merge(df_info, left_index=True, right_index=True, how='left')
        df.columns = df.columns.map(lambda x: x.lower())
        return df

    def hedge(self):
        pass