import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA



# 图表中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")  # 忽略一些警告 不影响运行



# ——————————————————————本地数据模块——————————————————————————————
# 读取数据
data = pd.read_csv("maotai_factor1.csv",encoding='gbk')
data = data.fillna(0)
data_date111=data[['date']]
# 数据表转换为浮点数
data['close'] = data['close'].str.replace(',', '').astype(float)
data['open'] = data['open'].str.replace(',', '').astype(float)
data['high'] = data['high'].str.replace(',', '').astype(float)
data['low'] = data['low'].str.replace(',', '').astype(float)
# 数据表转换为浮点数
data[['amt', 'MA5', 'MA10', 'MA20', 'DIF', 'DEA', 'MACD', 'K', 'D', 'J', 'RSI',
      'WR', 'WR1', 'BIAS1', 'BIAS2', 'BIAS3', 'PSY', 'PSYMA', 'CCI', 'ATR',
      'BBI']] = data[[ 'amt', 'MA5', 'MA10', 'MA20', 'DIF', 'DEA', 'MACD', 'K',
                       'D', 'J', 'RSI', 'WR', 'WR1', 'BIAS1', 'BIAS2', 'BIAS3',
                       'PSY', 'PSYMA', 'CCI', 'ATR', 'BBI']].astype(float)


# 全25特征
# data_x=data[['close', 'open', 'high', 'low','amt', 'MA5','MA10',
#                  'MA20','DIF','DEA','MACD','K','D','J','RSI','WR','WR1',
#                  'BIAS1','BIAS2','BIAS3','PSY','PSYMA','CCI','ATR','BBI',
#                  ]].values
# 4特征
data_x=data[['close','open', 'high', 'low']].values
data_x=np.array(data_x)

# ————————————————————————特征降维————————————————————————————
# 调用文件，用PCA、PCA获取降维后的特征
# processor = DataProcessor("maotai_factor.csv")

# data_pca = processor.get_data_pca()
# data_pca=np.array(data_pca)

# data_fa = processor.get_data_fa()
# data_fa=np.array(data_fa)

# data_x=data_pca
# data_factor=data_fa

# ——————————————————标准化——————————————————————
# minmax归一化
num_features = data_x.shape[1] # 获取特征数量
# 计算每个特征的最大值和最小值
min_values = data_x.min(axis=0) # 按列计算最小值
max_values = data_x.max(axis=0) # 按列计算最大值
# 进行归一化
data_x = (data_x - min_values) / (max_values - min_values)
max_values1 = max_values[0]
min_values1 = min_values[0]


# --------------------end--------------------------------------

# ——————————————————————————样本数据、标签数据处理—————————————————————————————————————
int_sequence_len = 5  # 一次输入多少天数据,即输入的时间步,可以调整,为序列长度
n_days = 5  # 设置要预测的天数
step_size = 1  # 滑动窗口的步长

int_a = data_x.shape[1] # 每个序列的长度,每个时间步数据的特征数量,即每个时间步的数据维度。固定的

train_x, train_y = [], []  # 样本和标签
for i in range(0, len(data_x) - int_sequence_len - n_days, step_size):
    train_x.append(data_x[i:i + int_sequence_len])
    # 获取当前样本后n_days天的股价值,并重复n_days次
    target_prices = [data_x[i + int_sequence_len + j] for j in range(n_days)]
    train_y.append(target_prices)


# 划分样本数据为训练集（70%）、验证集（20%）和测试集（10%），同时保持顺序不变
x_train, x_temp, y_train, y_temp = train_test_split(np.array(train_x), np.array(train_y), test_size=0.3, shuffle=False, random_state=1)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=1/3, shuffle=False, random_state=1)

# 提取最后一个维度的第一列
y_train = y_train[:, :, 0]
y_valid = y_valid[:, :, 0]
y_test = y_test[:, :, 0]

# 把列表类型转换为数组类型
x_train, y_train, x_valid, y_valid, x_test, y_test = np.array(x_train), np.array(y_train),np.array(x_valid),np.array(y_valid),np.array(x_test),np.array(y_test)
# ------end---------------


# 将三维张量转换为二维矩阵
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1] * x_valid.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
# ---------------end----------------

# # —————————————使用随机森林进行特征选择——————————————————
# feature_selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold='median')
# X_selected = feature_selector.fit_transform(x_train, y_train)
#
# # —————————————————进行主成分分析—————————————————
# pca = PCA(n_components=0.95)
# X_reduced = pca.fit_transform(x_train)


# -------------------end-------------------------

# 创建XGBoost模型
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1)
# 训练模型
model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=20, verbose=True)

'''
'max_depth': (2, 10),
'n_estimators': (100, 1000),
'learning_rate': (0.01, 0.3)
'''

# 在测试集上评估模型
predictions = model.predict(x_test)

# 反归一化
predictions = predictions * (max_values1 - min_values1) + min_values1
y_test = y_test * (max_values1 - min_values1) + min_values1

predictions = predictions[:, n_days-1, None]
y_test = y_test[:,n_days-1,None]

# 使用5折交叉验证评估模型
scores = cross_val_score(model, x_test, y_test, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE: {-scores.mean():.4f} (+/- {scores.std():.4f})")


# 计算 MSE
mse = mean_squared_error(y_test.flatten(), predictions.flatten())
print(f"Test MSE: {mse:.4f}")
# 计算其他评估指标
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
r2 = r2_score(y_test, predictions)

print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%, R2: {r2:.4f}')

# 绘制真实值与预测值对比图
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label='True')
plt.plot(range(len(predictions)), predictions, label='Predicted')
plt.legend()
plt.title('True vs Predicted Values')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()
