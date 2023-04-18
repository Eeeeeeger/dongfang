from data_utils import *


'''
升频
'''
# data = pd.read_excel('MacroData.xlsx', sheet_name='Growth', skiprows=1, index_col=0).iloc[:-1]
# data.columns = [columns_rename(x) for x in data.columns]
# reg_data = []
# for each in data.columns:
#     if each == 'GDP_不变价':
#         reg_data.append(preprocess(data[each], interpolate=True))
#     elif 'PMI' in each:
#         reg_data.append(preprocess(data[each], interpolate=True, dif=True, season=True))
#     else:
#         reg_data.append(preprocess(data[each], interpolate=True, season=True))
# reg_data = pd.concat(reg_data, axis=1)
# reg_data.columns = data.columns
#
# reg_data[reg_data.columns[-1]] = reg_data[reg_data.columns[-1]].shift(-1)
# growth_res = Parallel(n_jobs=len(reg_data.columns[:-1]))(
#     delayed(bootstrap_regression)(
#         reg_data,
#         reg_data.columns[-1],
#         each_col
#     )
#     for each_col in reg_data.columns[:-1]
# )
# growth_res = pd.concat(growth_res, axis=1)
# print(growth_res.T.sort_values('rsquared', ascending=False))

'''
降频
'''

data = pd.read_excel('MacroData.xlsx', sheet_name='Growth', skiprows=1, index_col=0).iloc[:-1]
data.columns = [columns_rename(x) for x in data.columns]
reg_data = []
# print(data.columns)
for each in data.columns:
    if each == 'GDP_不变价':
        reg_data.append(preprocess(data[each], interpolate=False))
    elif 'PMI' in each:
        reg_data.append(preprocess(data[each], interpolate=True, dif=True, season=True).resample(rule='Q').mean())
    else:
        reg_data.append(preprocess(data[each], interpolate=True, season=True).resample(rule='Q').mean())
reg_data = pd.concat(reg_data, axis=1)
reg_data.columns = data.columns

reg_data[reg_data.columns[-1]] = reg_data[reg_data.columns[-1]].shift(-1)
growth_res = Parallel(n_jobs=len(reg_data.columns[:-1]))(
    delayed(bootstrap_regression)(
        reg_data,
        reg_data.columns[-1],
        each_col
    )
    for each_col in reg_data.columns[:-1]
)
growth_res = pd.concat(growth_res, axis=1)
print(growth_res.T.sort_values('rsquared', ascending=False))