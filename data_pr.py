import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,SimpleRNN
import keras
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import os
import subprocess
import tensorflow as tf
import argparse
from tf_cfc import CfcCell, MixedCfcCell, LTCCell
import time
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

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

# 归一化处理，使用 MinMaxScaler
date_scaler = MinMaxScaler(feature_range=(0, 1))
data['normalized_timestamp'] = date_scaler.fit_transform(data['timestamp'].values.reshape(-1, 1))
# 打印归一化后的时间戳数组
data_date = np.array(data['normalized_timestamp']).reshape(-1, 1)


# 全25特征
data_x=data[['close', 'open', 'high', 'low','amt', 'MA5','MA10',
                 'MA20','DIF','DEA','MACD','K','D','J','RSI','WR','WR1',
                 'BIAS1','BIAS2','BIAS3','PSY','PSYMA','CCI','ATR','BBI',
                 ]].values
# 4特征
# data_x=data[['close','open', 'high', 'low']].values
data_x=np.array(data_x)

# z-score标准化
mean_values = []
std_values = []
length = data_x.shape[1]
for i in range(length):
    mean_data = np.mean(data_x[:, i])
    std_data = np.std(data_x[:, i])
    mean_values.append(mean_data)
    std_values.append(std_data)
    data_x[:, i] = (data_x[:, i] - mean_data) / std_data  # 标准化/归一化

# 将均值和标准差保存下来，用于反标准化
mean_values = np.array(mean_values)
std_values = np.array(std_values)

# print(data_x[:,0,None])

# 移动时间窗口参数，int_sequence_len为输入天数，n_days为输出天数
int_sequence_len = 16  # 序列长度，输入的时间步，即一次输入多少天数据,可以调整
int_a = data_x.shape[1] # 每个序列的长度，每个时间步数据的特征数量，即每个时间步的数据维度。固定的
n_days = 4  # 设置要预测的天数
step_size = 8  # 滑动窗口的步长

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

train_x= np.array(train_x)
train_x = train_x[:,:,0]

import random
# print(train_x.shape)
# exit()
# Randomly select 4 samples from train_x
random_samples = random.sample(range(len(train_x)), 4)

# Plot the randomly selected samples as images
plt.figure(figsize=(10,10),dpi=200)  # Adjust figure size to accommodate square images
for i, idx in enumerate(random_samples, 1):
    plt.subplot(2, 2, i)
    plt.imshow(train_x[idx].reshape(int(int_sequence_len / 4), int(step_size / 2)), cmap='gray',
               aspect='equal')  # Set aspect='equal'
    plt.title(f'Sample {i}', loc='right')
    plt.colorbar(label='Normalized Values')
    plt.xlabel('Step')
    plt.ylabel('Sequence')

    # Annotate with approximate timestamps
    start_date = train_date_x[idx][0][0]  # Assuming all sequences have the same start date
    end_date = train_date_y[idx][-1][0]  # Assuming all sequences have the same end date
    plt.annotate(f'Start: {start_date:.2f}\nEnd: {end_date:.2f}',
                 xy=(0.05, 0.02), xycoords='axes fraction',  # Adjust y-coordinate here (0.5, 0.2)
                 xytext=(1.5, 0.75), textcoords='axes fraction',
                 horizontalalignment='left', verticalalignment='center')
plt.tight_layout()
plt.show()


# # Plotting the data from data_x
# plt.figure(figsize=(12, 6))
# plt.plot(data_date, data_x[:,0,None])
# plt.title('Stock Data (Normalized)')
# plt.xlabel('Trading Days')
# plt.ylabel('Normalized Values')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# Plotting the data from data_x with transparent background grid and higher resolution
plt.figure(figsize=(16, 8), dpi=200)
plt.plot(data_date, data_x[:,0,None])
plt.title('Stock Data (Normalized)')
plt.xlabel('Trading Days', labelpad=10)  # Add padding to the x-axis label
plt.ylabel('Normalized Values', labelpad=10)  # Add padding to the y-axis label
plt.xticks(rotation=45)
plt.grid(alpha=0.5)  # Adjust transparency of the grid
plt.tight_layout()
plt.show()

