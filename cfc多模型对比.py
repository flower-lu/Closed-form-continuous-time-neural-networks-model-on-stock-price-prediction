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
import tf_cfc
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import binarize
from Factor_ans_use1 import DataProcessor
import tensorflow as tf
import argparse
from tf_cfc import CfcCell, MixedCfcCell, LTCCell
import time
import sys
# 调用模型评价指标
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import datetime
from scipy.interpolate import make_interp_spline

from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from keras.models import load_model
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator, AutoLocator
from datetime import datetime
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
import csv

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


# 全25特征
# data_x=data[['close', 'open', 'high', 'low','amt', 'MA5','MA10',
#                  'MA20','DIF','DEA','MACD','K','D','J','RSI','WR','WR1',
#                  'BIAS1','BIAS2','BIAS3','PSY','PSYMA','CCI','ATR','BBI',
#                  ]].values
# 4特征
data_x=data[['close','open', 'high', 'low']].values
data_x=np.array(data_x)

# 调用文件，用PCA、PCA获取降维后的特征
# processor = DataProcessor("maotai_factor.csv")

# data_pca = processor.get_data_pca()
# data_pca=np.array(data_pca)

# data_fa = processor.get_data_fa()
# data_fa=np.array(data_fa)

# data_x=data_pca
# data_factor=data_fa

# minmax归一化
num_features = data_x.shape[1] # 获取特征数量
# 计算每个特征的最大值和最小值
min_values = data_x.min(axis=0) # 按列计算最小值
max_values = data_x.max(axis=0) # 按列计算最大值
# 进行归一化
data_x = (data_x - min_values) / (max_values - min_values)

# --------------------end--------------------------------------

# ——————————————————————————样本数据标签数据处理—————————————————————————————————————
int_sequence_len = 10  # 一次输入多少天数据,即输入的时间步,可以调整,为序列长度
n_days = 1  # 设置要预测的天数
step_size = 1  # 滑动窗口的步长

int_a = data_x.shape[1] # 每个序列的长度,每个时间步数据的特征数量,即每个时间步的数据维度。固定的

train_x, train_y = [], []  # 样本和标签
for i in range(0, len(data_x) - int_sequence_len - n_days, step_size):
    train_x.append(data_x[i:i + int_sequence_len])
    # 获取当前样本后n_days天的股价值,并重复n_days次
    target_prices = [data_x[i + int_sequence_len + j] for j in range(n_days)]
    train_y.append(target_prices)

train_date_x, train_date_y = [], []  # 日期样本和标签
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

# 日期反归一化
original_timestamps = y_date_test * (timestamp_max_param - timestamp_min_param) + timestamp_min_param
# 取整
original_timestamps = original_timestamps.astype(int)
y_date_re = [datetime.fromtimestamp(ts) for ts in original_timestamps.flat]

# ——————————————————————选择不同的神经网络————————————————————————————————————————
# 使用LSTM
# model0 = Sequential()
# model0.add(LSTM(100, input_shape=(int_sequence_len, int_a), return_sequences=True, activation='relu'))
# model0.add(LSTM(10, return_sequences=False, activation='relu'))
# model0.add(Dense(units=10, activation='relu'))
# model0.add(Dense(1))
#
# # 对模型进行编译，选用Adam优化器，学习率为0.01
# model0.compile(optimizer=keras.optimizers.Adam(0.01), loss='mean_squared_error')
# model0.summary()
# # 将训练集和测试集放入网络进行训练，每批次送入的数据为512个数据，一共训练30轮，将测试集样本放入到神经网络中测试其验证集的loss值
# history = model0.fit(x_train, y_train, batch_size=512, epochs=30, verbose=2, validation_data=(x_valid, y_valid))
# # 保存训练好的模型
# model0.save('model0.h5')
#
# # 使用简单RNN
# model1 = Sequential()
# model1.add(SimpleRNN(196, input_shape=(int_sequence_len, int_a), activation='tanh', return_sequences=True))
# model1.add(Dropout(0.2))
# model1.add(SimpleRNN(64, activation='relu'))
# model1.add(Dense(units=32, activation='relu'))
# model1.add(Dense(1))
#
# # 对模型进行编译，选用Adam优化器，学习率为0.01
# model1.compile(optimizer=keras.optimizers.RMSprop(0.001), loss='mean_squared_error')
# model1.summary()
# # 将训练集和测试集放入网络进行训练，每批次送入的数据为512个数据，一共训练30轮，将测试集样本放入到神经网络中测试其验证集的loss值
# history1 = model1.fit(x_train, y_train, batch_size=512, epochs=30, verbose=2, validation_data=(x_valid, y_valid))
# # 保存训练好的模型
# model1.save('model1.h5')
#
# #使用GRU
# model2 = Sequential()
# model2.add(GRU(128, input_shape=(int_sequence_len, int_a), activation='tanh', return_sequences=True))
# model2.add(Dropout(0.2))
# model2.add(GRU(64, activation='tanh', return_sequences=False))
# model2.add(Dropout(0.2))
# model2.add(Dense(units=32, activation='tanh'))
# model2.add(Dropout(0.2))
# model2.add(Dense(1))
#
# model2.compile(optimizer=keras.optimizers.RMSprop(0.0001), loss='mean_squared_error')
# model2.summary()
# # 将训练集和测试集放入网络进行训练，每批次送入的数据为512个数据，一共训练30轮，将测试集样本放入到神经网络中测试其验证集的loss值
# history2 = model2.fit(x_train, y_train, batch_size=256, epochs=30, verbose=2, validation_data=(x_valid, y_valid))
# # 保存训练好的模型
# model2.save('model2.h5')
#
# # 使用1D CNN
# model3 = Sequential()
# model3.add(Conv1D(filters=64, kernel_size=3, activation='tanh', input_shape=(int_sequence_len, int_a)))
# model3.add(MaxPooling1D(pool_size=2))
# model3.add(Flatten())
# model3.add(Dense(units=32, activation='relu'))
# model3.add(Dropout(0.2))
# model3.add(Dense(1))
#
# model3.compile(optimizer=keras.optimizers.RMSprop(0.001), loss='mean_squared_error')
# model3.summary()
# # 将训练集和测试集放入网络进行训练，每批次送入的数据为512个数据，一共训练30轮，将测试集样本放入到神经网络中测试其验证集的loss值
# history3 = model3.fit(x_train, y_train, batch_size=64, epochs=30, verbose=2, validation_data=(x_valid, y_valid))
# # 保存训练好的模型
# model3.save('model3.h5')
# # -----------------end----------
#
# # ————————————————————————测试过程————————————————————————————————————
# # 导入训练好的模型参数
# model0 = load_model("model0.h5")
# model1 = load_model("model1.h5")
# model2 = load_model("model2.h5")
# model3 = load_model("model3.h5")
#
# # 测试集输入模型进行预测
# predicted = model0.predict(x_test)
# predicted1 = model1.predict(x_test)
# predicted2 = model2.predict(x_test)
# predicted3 = model3.predict(x_test)
#
# # 反归一化
# prediction = predicted * (max_values - min_values) + min_values
# prediction1 = predicted1 * (max_values - min_values) + min_values
# prediction2 = predicted2 * (max_values - min_values) + min_values
# prediction3 = predicted3 * (max_values - min_values) + min_values
#
# real = y_test * (max_values - min_values) + min_values
#
# # 提取第一列数据，即为收盘价
# prediction = prediction[:, 0, None]
# prediction1 = prediction1[:, 0, None]
# prediction2 = prediction2[:, 0, None]
# prediction3 = prediction3[:, 0, None]
#
# real = real[:, 0, None]
#
#
# # 指定要保存的文件名
# csv_filename = "predictions3.csv"
#
# # 将数据组合成一个列表
# data_output = [
#     ["Prediction", "Prediction1", "Prediction2", "Prediction3", "Real"],
#     # 将每一行的数据组合成一个列表
#     *zip(prediction.flatten(), prediction1.flatten(), prediction2.flatten(), prediction3.flatten(), real.flatten())
# ]
#
# # 将数据写入 CSV 文件
# with open(csv_filename, "w", newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerows(data_output)
#
# print(f"数据已保存到文件: {csv_filename}")
#
# # 计算模型的评价指标
# R2 = r2_score(real, prediction)
# MAE = mean_absolute_error(real, prediction)
# RMSE = np.sqrt(mean_squared_error(real, prediction))
# MAPE = np.mean(np.abs((real-prediction) / prediction))
#
# # 计算评价指标
# R2_1 = r2_score(real, prediction1)
# MAE_1 = mean_absolute_error(real, prediction1)
# RMSE_1 = np.sqrt(mean_squared_error(real, prediction1))
# MAPE_1 = np.mean(np.abs((real-prediction1) / prediction1))
#
# R2_2 = r2_score(real, prediction2)
# MAE_2 = mean_absolute_error(real, prediction2)
# RMSE_2 = np.sqrt(mean_squared_error(real, prediction2))
# MAPE_2 = np.mean(np.abs((real-prediction2) / prediction2))
#
# R2_3 = r2_score(real, prediction3)
# MAE_3 = mean_absolute_error(real, prediction3)
# RMSE_3 = np.sqrt(mean_squared_error(real, prediction3))
# MAPE_3 = np.mean(np.abs((real-prediction3) / prediction3))
#
# # 打印模型的评价指标
# print('R2:', R2)
# print('MAE:', MAE)
# print('RMSE:', RMSE)
# print('MAPE:', MAPE)
#
# print('For prediction 1:')
# print('R2:', R2_1)
# print('MAE:', MAE_1)
# print('RMSE:', RMSE_1)
# print('MAPE:', MAPE_1)
#
# print('For prediction 2:')
# print('R2:', R2_2)
# print('MAE:', MAE_2)
# print('RMSE:', RMSE_2)
# print('MAPE:', MAPE_2)
#
# print('For prediction 3:')
# print('R2:', R2_3)
# print('MAE:', MAE_3)
# print('RMSE:', RMSE_3)
# print('MAPE:', MAPE_3)
#
# exit()

# 创建一个空列表来存储预测值
predicted_values_list = []
# ————————————————————————————构建cfc模型————————————————————————————
class BackupCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(BackupCallback, self).__init__()
        self.saved_weights = None
        self.model = model
        self.best_loss = np.PINF

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best_loss:
            self.best_loss = logs["val_loss"]
            # print(f" new best -> {logs['val_loss']:0.3f}")
            self.saved_weights = self.model.get_weights()

    def restore(self):
        if self.best_loss is not None:
            self.model.set_weights(self.saved_weights)

def eval(config, index_arg, verbose=2):

    if config.get("use_ltc"):
        cell = LTCCell(units=config["size"])
    elif config["use_mixed"]:
        cell = MixedCfcCell(units=config["size"], hparams=config)
    else:
        cell = CfcCell(units=config["size"], hparams=config)
    pixel_input = tf.keras.Input(shape=(int_sequence_len, int_a), name="input")
    time_input = tf.keras.Input(shape=(int_sequence_len, 1), name="time")

    # 用的是RNN架构
    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
    # dense_layer = tf.keras.layers.Dense(1)
    # output_states = rnn((pixel_input, time_input))
    # y = dense_layer(output_states)

    output_units = n_days  # n天的维度
    output_layer = tf.keras.layers.Dense(output_units)
    output_states = rnn((pixel_input, time_input))
    y = output_layer(output_states)

    model = tf.keras.Model(inputs=[pixel_input, time_input], outputs=[y])
    model.summary()

    # 编译模型
    base_lr = config["base_lr"]
    decay_lr = config["decay_lr"]
    train_steps = x_train.shape[0] // config["batch_size"]
    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, train_steps, decay_lr
    )
    opt = (
        tf.keras.optimizers.Adam
        if config["optimizer"] == "adam"
        else tf.keras.optimizers.RMSprop
    )
    optimizer = opt(learning_rate_fn, clipnorm=config["clipnorm"])
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
    )
    start_time = time.time()
    # 训练模型
    history = model.fit(
        x=(x_train, x_date_train),
        y=y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=((x_valid, x_date_valid), y_valid),
        callbacks=[BackupCallback(model)],
        verbose=2,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_minutes = elapsed_time / 60.0
    print(f"Total time taken for training: {elapsed_time_minutes:.4f} minutes")


    # 保存训练好的模型
    # model_path = 'Cfc_model.h5'
    # model.save(model_path, include_optimizer=False)

    # ——————————————————————训练过程——————————————————————————————————————————

    # 使用训练好的模型对测试数据进行预测
    predicted_values = model.predict((x_test, x_date_test))
    true_values = y_test  # y_train代表预测后相对应的标签，选取第一列特征即为收盘价


    predicted_values = predicted_values[:, n_days-1, None]  # 有n天的预测值，我需要选取预测的第几天的数据
    true_values = true_values[:, n_days-1, None]  # 有n天的预测值，我需要选取预测的第几天的数据

    # 反归一化
    predicted_values = predicted_values * (max_values - min_values) + min_values
    true_values = true_values * (max_values - min_values) + min_values

    # 提取第一列数据，即为收盘价
    predicted_values = predicted_values[:,0,None]
    true_values = true_values[:,0,None]
    # 将预测值添加到列表中
    predicted_values_list.append(predicted_values)

    # 评估模型在测试集或验证集上的性能，并返回评估结果。
    # test_loss = model.evaluate(
    #     x=(x_valid, x_date_valid),
    #     y=y_valid,
    #     verbose=2,
    # )
    test_loss = model.evaluate(
        x=(x_test, x_date_test),
        y=y_test,
        verbose=2,
    )

    # 计算模型的评价指标(标准化后)
    R2 = r2_score(true_values, predicted_values)
    MAE = mean_absolute_error(true_values, predicted_values)
    RMSE = np.sqrt(mean_squared_error(true_values, predicted_values))
    MAPE = np.mean(np.abs((true_values-predicted_values) / predicted_values))

    # 打印模型的评价指标
    print('R2:', R2)
    print('MAE:', MAE)
    print('RMSE:', RMSE)
    print('MAPE:', MAPE)

    # 画图
    # plt.figure(figsize=(12, 8))
    # plt.plot(y_date_re, prediction, label='LSTM')
    # plt.plot(y_date_re, prediction1, label='RNN')
    # plt.plot(y_date_re, prediction2, label='GRU')
    # plt.plot(y_date_re, prediction3, label='CNN')
    # plt.plot(y_date_re, predicted_values, label='CfC1')
    # plt.plot(y_date_re, real, label='真实值')
    # # 设置主刻度的格式
    # ax = plt.gca()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # # 设置次刻度
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # # 旋转x轴刻度
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(fontsize=15)
    # plt.legend(loc='best', fontsize=15)
    # plt.ylabel('股价', fontsize=15)
    # plt.xlabel('时间', fontsize=15)
    # plt.title("股价预测", fontsize=15)
    # plt.show()

    return test_loss,history
# -----------------------end--------------------------------------

# 默认参数设置
BEST_DEFAULT = {
    "clipnorm": 1,
    "optimizer": "Rmsprop",
    "batch_size": 64,
    "size": 512,
    "epochs": 50,
    "base_lr": 0.0004,
    "decay_lr": 0.95,
    "backbone_activation": "relu",
    "backbone_dr": 0,
    "forget_bias": 0.8,
    "backbone_units": 256,
    "backbone_layers": 1,
    "weight_decay": 3e-06,
    "use_mixed": False,
}
# MSE: 0.61654 +- 0.00634
BEST_MIXED = {
    "clipnorm": 10,
    "optimizer": "Rmsprop",
    "batch_size": 64,
    "size": 512,
    "epochs": 50,
    "base_lr": 0.0004,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.2,
    "forget_bias": 2.1,
    "backbone_units": 256,
    "backbone_layers": 1,
    "weight_decay": 3e-06,
    "use_mixed": True,
    "no_gate": False,
}
# 0.65040 $\pm$ 0.00814
BEST_NO_GATE = {
    "clipnorm": 1,
    "optimizer": "Rmsprop",
    "batch_size": 512,
    "size": 512,
    "epochs": 50,
    "base_lr": 0.0004,
    "decay_lr": 0.95,
    "backbone_activation": "relu",
    "backbone_dr": 0,
    "forget_bias": 0.8,
    "backbone_units": 256,
    "backbone_layers": 1,
    "weight_decay": 3e-06,
    "use_mixed": False,
    "no_gate": True,
}
# 0.94844 $\pm$ 0.00988
BEST_MINIMAL = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 512,
    "size": 256,
    "epochs": 50,
    "base_lr": 0.006,
    "decay_lr": 0.95,
    "backbone_activation": "silu",
    "backbone_dr": 0.0,
    "forget_bias": 5.0,
    "backbone_units": 192,
    "backbone_layers": 1,
    "weight_decay": 1e-06,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
}
# 0.66225 $\pm$ 0.01330
BEST_LTC = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 512,
    "size": 128,
    "epochs": 50,
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.0,
    "forget_bias": 2.4,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-05,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
}

# 综合评估结果
def score(config):
    losses = []
    histories = []
    for i in range(5):
        loss, history = eval(config, i)
        losses.append(loss)
        histories.append(history)

    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    print(f"MSE: {mean_loss:0.5f} +/- {std_loss:0.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # score(BEST_DEFAULT)
    # score(BEST_MIXED)
    # score(BEST_NO_GATE)
    # score(BEST_MINIMAL)
    # score(BEST_LTC)

    parser.add_argument("--use_mixed", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--use_ltc", action="store_true")

    args = parser.parse_args()

    if args.minimal:
        score(BEST_MINIMAL)
    elif args.no_gate:
        score(BEST_NO_GATE)
    elif args.use_ltc:
        score(BEST_LTC)
    elif args.use_mixed:
        score(BEST_MIXED)
    else:
        score(BEST_DEFAULT)


    # 将预测值列表写入CSV文件
    with open('predicted_values2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for values in zip(*predicted_values_list):
            writer.writerow(values)



