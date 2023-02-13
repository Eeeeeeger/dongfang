import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # 导入设置坐标轴的模块
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.style.use('seaborn')  # plt.style.use('dark_background')

df = pd.read_excel("国债期货跨期价差.xlsx", index_col=0, sheet_name='Sheet3')
df.index = pd.to_datetime(df.index)
df = df.loc['2018-01-01':]
fig, ax1 = plt.subplots(1, 1, figsize=(20, 8))
# 绘制累计收益曲线
ax2 = ax1.twinx()
ax1.yaxis.set_ticks_position('right')
ax2.yaxis.set_ticks_position('left')
(df['dif1']).plot(ax=ax1, label='price diff (right)', rot=0, alpha=0.5, fontsize=13, grid=False, color='deepskyblue')
# 绘制10年国债收益率
(df['dif2']).plot(ax=ax1, label='bond yield (right)', rot=0, alpha=0.5, fontsize=13, color='seagreen', grid=False)
# ax2.bar(df[df['tag'] == 1].index, height=1, width=1, color='lightcoral',label='Contract change')
# ax2.set_xbound(lower=df['GB10'].min(), upper=df['GB10'].max())
df['tag'].plot(ax=ax2, label='Contract change', rot=0, alpha=0.5, fontsize=13,)
# ax1.set_xbound(lower=df['价差'].min(), upper=df['价差'].max())
# 主轴定位器：每 5 个月显示一个日期：根据具体天数来做排版
# ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
# 同时绘制双轴的图例
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend(h1 + h2, l1 + l2, fontsize=15, loc='upper left', ncol=1)
fig.tight_layout()
plt.savefig('./basis.png')

df['distance'] = float("nan")
for ii in range(len(df)):
    try:
        df['distance'].iloc[ii] = min([(x-df.index[ii]).days for x in df[df['tag']==1].index if (x-df.index[ii]).days<=50 and (x-df.index[ii]).days>=0])
    except:
        pass

fig, ax1 = plt.subplots(1, 1, figsize=(20, 8))
dif1 = df.dropna().groupby('distance')['dif1']
dif2 = df.dropna().groupby('distance')['dif2']
dif1.mean()[dif1.count()>=5].plot()
dif2.mean()[dif2.count()>=5].plot()
plt.legend()
plt.gca().invert_xaxis()
plt.savefig('./basis2.png')

