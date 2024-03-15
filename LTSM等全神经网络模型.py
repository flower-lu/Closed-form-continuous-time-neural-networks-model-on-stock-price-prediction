import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,SimpleRNN
import keras
import numpy as np
import warnings
import os
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, GRU, LSTM , Dropout
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten
import keras.optimizers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator, AutoLocator
from datetime import datetime

# 环境问题处理
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# 图表中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")  # 忽略一些警告 不影响运行

# 读取数据
data = pd.read_csv("shangzheng_factor.csv",encoding='gbk')
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


# ——————————————特征处理————————————————————————————————————————————————
# 将日期列转换为日期时间类型
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
# 将日期时间转换为时间戳（以秒为单位）
data['timestamp'] = data['date'].apply(lambda x: x.timestamp()).astype(int)
# print(data['timestamp'])

# 假设data['timestamp']是你的原始时间戳数据,是一个一维数组
timestamps = data['timestamp'].values
# 找到时间戳的最小值和最大值
timestamp_min = np.min(timestamps)
timestamp_max = np.max(timestamps)
# 进行归一化
normalized_timestamps = (timestamps - timestamp_min) / (timestamp_max - timestamp_min)
# 保存用于反归一化的参数
timestamp_min_param = timestamp_min
timestamp_max_param = timestamp_max
# 将归一化后的时间戳重新赋值给data
data['normalized_timestamp'] = normalized_timestamps
data_date = np.array(data['normalized_timestamp']).reshape(-1, 1)


# # 归一化处理，使用 MinMaxScaler
# date_scaler = MinMaxScaler(feature_range=(0, 1))
# # 保存min和max
# data_min = date_scaler.data_range_[0]
# data_max = date_scaler.data_range_[1]
# data['normalized_timestamp'] = date_scaler.fit_transform(data['timestamp'].values.reshape(-1, 1))
# # 打印归一化后的时间戳数组
# data_date = np.array(data['normalized_timestamp']).reshape(-1, 1)

# 全25特征
# data_x=data[['close', 'open', 'high', 'low','amt', 'MA5','MA10',
#                  'MA20','DIF','DEA','MACD','K','D','J','RSI','WR','WR1',
#                  'BIAS1','BIAS2','BIAS3','PSY','PSYMA','CCI','ATR','BBI',
#                  ]].values
# 4特征
data_x=data[['close','open', 'high', 'low']].values
data_x=np.array(data_x)

# 对数据集进行minmax归一化
num_features = data_x.shape[1] # 获取特征数量
# 计算每个特征的最大值和最小值
min_values = data_x.min(axis=0) # 按列计算最小值
max_values = data_x.max(axis=0) # 按列计算最大值
# 进行归一化
data_x = (data_x - min_values) / (max_values - min_values)

# 移动时间窗口参数，int_sequence_len为输入天数，n_days为输出天数
int_sequence_len = 20  # 一次输入多少天数据，即输入的时间步，可以调整，为序列长度
n_days = 1  # 设置要预测的天数
step_size = 1  # 滑动窗口的步长

int_a = data_x.shape[1] # 每个序列的长度，每个时间步数据的特征数量，即每个时间步的数据维度。固定的

train_x, train_y = [], []  # 训练集
for i in range(0, len(data_x) - int_sequence_len - n_days, step_size):
    train_x.append(data_x[i:i + int_sequence_len])
    # 获取当前样本后n_days天的股价值，并重复n_days次
    target_prices = [data_x[i + int_sequence_len + j] for j in range(n_days)]
    train_y.append(target_prices)

train_date_x, train_date_y = [], []  # 日期训练集
for i in range(0, len(data_x)-int_sequence_len-n_days, step_size):
    train_date_x.append(data_date[i : i+int_sequence_len])
    target_dates = [data_date[i + int_sequence_len + j] for j in range(n_days)]
    train_date_y.append(target_dates)


# 划分样本数据为训练集（70%）、验证集（20%）和测试集（10%），同时保持顺序不变
x_train, x_temp, y_train, y_temp = train_test_split(np.array(train_x), np.array(train_y), test_size=0.3, shuffle=False, random_state=1)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=1/3, shuffle=False, random_state=1)

# 划分日期特征数据，同样保持顺序不变
x_date_train, x_date_temp, y_date_train, y_date_temp = train_test_split(np.array(train_date_x), np.array(train_date_y), test_size=0.3, shuffle=False, random_state=1)
x_date_valid, x_date_test, y_date_valid, y_date_test = train_test_split(x_date_temp, y_date_temp, test_size=1/3, shuffle=False, random_state=1)

# 提取最后一个维度的第一列
y_train = y_train[:, :, 0]
y_valid = y_valid[:, :, 0]
y_test = y_test[:, :, 0]

# 把df类型转换为数组类型
x_train, y_train, x_valid, y_valid, x_test, y_test = np.array(x_train), np.array(y_train),np.array(x_valid),np.array(y_valid),np.array(x_test),np.array(y_test)


# ——————————————————————选择不同的神经网络————————————————————————————————————————
# 使用LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(int_sequence_len, int_a), return_sequences=True, activation='relu'))
model.add(LSTM(10, return_sequences=False, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(1))

# 对模型进行编译，选用Adam优化器，学习率为0.01
model.compile(optimizer=keras.optimizers.Adam(0.01), loss='mean_squared_error')
model.summary()
# 将训练集和测试集放入网络进行训练，每批次送入的数据为512个数据，一共训练30轮，将测试集样本放入到神经网络中测试其验证集的loss值
history = model.fit(x_train, y_train, batch_size=512, epochs=30, verbose=2, validation_data=(x_valid, y_valid))
# 保存训练好的模型
model.save('model0.h5')

# 使用简单RNN
model1 = Sequential()
model1.add(SimpleRNN(196, input_shape=(int_sequence_len, int_a), activation='tanh', return_sequences=True))
model1.add(SimpleRNN(64, activation='relu'))
model1.add(Dense(units=32, activation='relu'))
model1.add(Dense(1))

# 对模型进行编译，选用Adam优化器，学习率为0.01
model1.compile(optimizer=keras.optimizers.RMSprop(0.001), loss='mean_squared_error')
model1.summary()
# 将训练集和测试集放入网络进行训练，每批次送入的数据为512个数据，一共训练30轮，将测试集样本放入到神经网络中测试其验证集的loss值
history1 = model1.fit(x_train, y_train, batch_size=512, epochs=30, verbose=2, validation_data=(x_valid, y_valid))
# 保存训练好的模型
model1.save('model1.h5')

#使用GRU
model2 = Sequential()
model2.add(GRU(128, input_shape=(int_sequence_len, int_a), activation='tanh', return_sequences=True))
model2.add(Dropout(0.2))
model2.add(GRU(64, activation='tanh', return_sequences=False))
model2.add(Dropout(0.2))
model2.add(Dense(units=32, activation='tanh'))
model2.add(Dropout(0.2))
model2.add(Dense(1))

model2.compile(optimizer=keras.optimizers.RMSprop(0.0001), loss='mean_squared_error')
model2.summary()
# 将训练集和测试集放入网络进行训练，每批次送入的数据为512个数据，一共训练30轮，将测试集样本放入到神经网络中测试其验证集的loss值
history2 = model2.fit(x_train, y_train, batch_size=256, epochs=30, verbose=2, validation_data=(x_valid, y_valid))
# 保存训练好的模型
model2.save('model2.h5')

# 使用1D CNN
model3 = Sequential()
model3.add(Conv1D(filters=64, kernel_size=3, activation='tanh', input_shape=(int_sequence_len, int_a)))
model3.add(MaxPooling1D(pool_size=2))
model3.add(Flatten())
model3.add(Dense(units=32, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(1))

model3.compile(optimizer=keras.optimizers.RMSprop(0.001), loss='mean_squared_error')
model3.summary()
# 将训练集和测试集放入网络进行训练，每批次送入的数据为512个数据，一共训练30轮，将测试集样本放入到神经网络中测试其验证集的loss值
history3 = model3.fit(x_train, y_train, batch_size=64, epochs=30, verbose=2, validation_data=(x_valid, y_valid))
# 保存训练好的模型
model3.save('model3.h5')
# -----------------end---------------------

# 记录训练损失
# training_loss = history.history['loss']
# test_loss = history.history['val_loss']
# # 创建迭代数量
# epoch_count = range(1, len(training_loss) + 1)
# # 可视化损失历史
# plt.plot(epoch_count, training_loss, 'r--')
# plt.plot(epoch_count, test_loss, 'b-')
# plt.legend(['Training Loss', 'Test Loss'])
# plt.title("Epoch-loss")
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()
#
# # 绘制训练集和测试集的loss值对比图
# # 创建一个大小为（12，8）的画布
# plt.figure(figsize=(12, 8))
#
# # 传入训练集的loss和验证集的loss
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='val')
#
# # 设置图的参数，设置图的名字
# plt.title("LSTM神经网络loss值", fontsize=15)
#
# # 设置xy轴的刻度值大小
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
#
# # 设置xy轴的标签
# plt.ylabel('loss值', fontsize=15)
# plt.xlabel('训练轮次', fontsize=15)
#
# # 设置图例文字大小
# plt.legend(fontsize=15)
# plt.show()
# ------------------end-------------------------------------


# ————————————————————————测试过程————————————————————————————————————
# 导入训练好的模型参数
model = load_model("model0.h5")
model1 = load_model("model1.h5")
model2 = load_model("model2.h5")
model3 = load_model("model3.h5")

# 测试集输入模型进行预测
predicted = model.predict(x_test)
predicted1 = model1.predict(x_test)
predicted2 = model2.predict(x_test)
predicted3 = model3.predict(x_test)

# 反归一化
prediction = predicted * (max_values - min_values) + min_values
prediction1 = predicted1 * (max_values - min_values) + min_values
prediction2 = predicted2 * (max_values - min_values) + min_values
prediction3 = predicted3 * (max_values - min_values) + min_values

real = y_test * (max_values - min_values) + min_values

# 提取第一列数据，即为收盘价
prediction = prediction[:, 0, None]
prediction1 = prediction1[:, 0, None]
prediction2 = prediction2[:, 0, None]
prediction3 = prediction3[:, 0, None]

real = real[:, 0, None]

# 计算模型的评价指标
# R2 = r2_score(real, prediction)
# MAE = mean_absolute_error(real, prediction)
# RMSE = np.sqrt(mean_squared_error(real, prediction))
# MAPE = np.mean(np.abs((real-prediction) / prediction))
# # 打印模型的评价指标
# print('R2:', R2)
# print('MAE:', MAE)
# print('RMSE:', RMSE)
# print('MAPE:', MAPE)


# 绘制真实值和预测值对比图
# 创建一个大小为（12，8）的画布
# plt.figure(figsize=(12, 8))
# # 传入预测值和真实值
# plt.plot(prediction, label='预测值')
# plt.plot(real, label='真实值')
# # 设置xy轴的刻度值大小
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# # 设置xy轴的标签
# plt.legend(loc='best', fontsize=15)
# plt.ylabel('股价', fontsize=15)
# plt.xlabel('时间', fontsize=15)
# # 设置图的参数，设置图的名字
# plt.title("股价预测", fontsize=15)
# plt.show()

# len_ = [i for i in range(len(y_test))]
# plt.xlabel('标签', fontsize=8)
# plt.ylabel('均值', fontsize=8)
# plt.plot(len_, real, label='real', color="blue")
# plt.plot(len_, prediction, label='prediction', color="yellow")
# plt.title("最低气温预测走势图")
# plt.show()
# plt.clf()

# 反归一化
original_timestamps = y_date_test * (timestamp_max_param - timestamp_min_param) + timestamp_min_param
# 取整
original_timestamps = original_timestamps.astype(int)
# 转换为datetime对象
y_date_test = [datetime.fromtimestamp(ts) for ts in original_timestamps.flat]

plt.figure(figsize=(12, 8))
plt.plot(y_date_test, prediction, label='预测值0')
plt.plot(y_date_test, prediction1, label='预测值1')
plt.plot(y_date_test, prediction2, label='预测值2')
plt.plot(y_date_test, prediction3, label='预测值3')

plt.plot(y_date_test, real, label='真实值')
# 设置主刻度的格式
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# 设置次刻度
ax.xaxis.set_minor_locator(AutoMinorLocator())
# 旋转x轴刻度
plt.xticks(rotation=45, ha='right')
plt.yticks(fontsize=15)
plt.legend(loc='best', fontsize=15)
plt.ylabel('股价', fontsize=15)
plt.xlabel('时间', fontsize=15)
plt.title("股价预测", fontsize=15)
plt.show()

# -------end-------------------------------------------------------

# model = load_model("LSTM_model.h5")
# model_jiazai_1 = model
# # 加载保存好的模型
# result = []
# y1_pred_lstm = model_jiazai_1.predict(x_test)
# for i in range(len(y1_pred_lstm)):
#     print("真实：", y_test[i])
#     print("预测：", y1_pred_lstm[i])
#     result.append([y_test[i][0], y1_pred_lstm[i][0]])
#     print("-----------------------")
#
#
# # 所有画图
# len_ = [i for i in range(len(y_test))]
# plt.xlabel('标签', fontsize=8)
# plt.ylabel('均值', fontsize=8)
# plt.plot(len_, y_test, label='y_test', color="blue")
# plt.plot(len_, y1_pred_lstm, label='y1_pred_lstm', color="yellow")
# plt.title("最低气温预测走势图")
# plt.show()
# plt.clf()



