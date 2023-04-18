from DataGenerator import DataGenerator
from RollingPredictor import RollingPredictor
import warnings

warnings.filterwarnings('ignore')
# 绘制图形
data = DataGenerator()
data.calc_derivative_factors()
data.calc_label(derivative=True)
features = data.features
df = data.derivative_df


'''
PARAMS FINE-TUNING PART IN TRAIN DATASET
'''
#
# a = RollingPredictor(df, features)
# a.fitting('LinearRegression')
# a.predict()
#
# a.fitting('LogisticRegression')
# a.predict()

a = RollingPredictor(df, features, train_period_lt=[500, 750, 1000, 1250], rolling_period_lt=[120, 250, 500])
a.fitting('LgbRegressor')
a.predict()