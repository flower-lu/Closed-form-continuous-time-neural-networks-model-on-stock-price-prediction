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
import tushare as ts
from datetime import datetime, date, timedelta

import tensorflow.keras.backend as K

import json
# 环境问题处理
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# 图表中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")  # 忽略一些警告 不影响运行

from tensorflow.keras import mixed_precision

# 设置策略为混合精度
# mixed_precision.set_global_policy('mixed_float16')


# 设置允许TensorFlow动态增长GPU内存使用，这可以避免在GPU显存不足时失败,但需要格外注意内存管理。
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ——————————————————————————tushare模块————————————————————————————————
# pro = ts.pro_api('647f6840944a4425d46c97c08cf20af6b656bb79673bd1635ebdf0ce')
# # 获取始末时间
# begin = '20120101'
# today = datetime.today().date()
# dayofWeek = today.weekday()
#
# if dayofWeek == 6:  # 星期日
#     end = (today - timedelta(days=3)).strftime('%Y%m%d')
#     end_ = (today - timedelta(days=3)).strftime('%Y-%m-%d')
# elif dayofWeek == 5:  # 星期六
#     end = (today - timedelta(days=2)).strftime('%Y%m%d')
#     end_ = (today - timedelta(days=2)).strftime('%Y-%m-%d')
# else:
#     end = (today - timedelta(days=1)).strftime('%Y%m%d')
#     end_ = (today - timedelta(days=1)).strftime('%Y-%m-%d')
#
# print(f"end: {end}, end_: {end_}")
#
# # 获取这段时间里面的股票价格
# data=pro.query('daily', ts_code='000681.SZ', start_date=begin, end_date=end) #放假期间股票停止交易
# df = pd.DataFrame(data)
# df.to_csv('SJZG000681.csv')
# Stock_XRHJ = pd.read_csv('SJZG000681.csv',index_col = 'trade_date',parse_dates=['trade_date'])
# df = pd.DataFrame(Stock_XRHJ)
# df=df.iloc[::-1]
# data = df
# data = data.reset_index()

# -----------------------end----------------------------

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

# --------------------------------end--------------------------

# ——————————————日期处理————————————————————————————————————————————————
# 将日期列转换为日期时间类型
# data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')  # 用tushare数据时采用
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
# 将日期时间转换为时间戳（以秒为单位）
# data['timestamp'] = data['trade_date'].apply(lambda x: x.timestamp()).astype(int)
data['timestamp'] = data['date'].apply(lambda x: x.timestamp()).astype(int)
# print(data['timestamp'])

# 归一化处理，使用 MinMaxScaler
date_scaler = MinMaxScaler(feature_range=(0, 1))
data['normalized_timestamp'] = date_scaler.fit_transform(data['timestamp'].values.reshape(-1, 1))
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

# ——————————————————————————————选取特征——————————————————————————
# 全25特征
data_x=data[['close', 'open', 'high', 'low','amt', 'MA5','MA10',
                 'MA20','DIF','DEA','MACD','K','D','J','RSI','WR','WR1',
                 'BIAS1','BIAS2','BIAS3','PSY','PSYMA','CCI','ATR','BBI',
                 ]].values
# 4特征
# data_x=data[['close','open', 'high', 'low']].values
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

# --------------------end--------------------------------------

# ——————————————————————————样本数据、标签数据处理—————————————————————————————————————
int_sequence_len = 5  # 一次输入多少天数据,即输入的时间步,可以调整,为序列长度
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

# 把列表类型转换为数组类型
x_train, y_train, x_valid, y_valid, x_test, y_test = np.array(x_train), np.array(y_train),np.array(x_valid),np.array(y_valid),np.array(x_test),np.array(y_test)


# 日期反归一化
original_timestamps_train = y_date_train * (timestamp_max_param - timestamp_min_param) + timestamp_min_param
original_timestamps_valid = y_date_valid * (timestamp_max_param - timestamp_min_param) + timestamp_min_param
original_timestamps_test = y_date_test * (timestamp_max_param - timestamp_min_param) + timestamp_min_param

# 取整
original_timestamps_train = original_timestamps_train.astype(int)
original_timestamps_valid = original_timestamps_valid.astype(int)
original_timestamps_test = original_timestamps_test.astype(int)

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
        metrics=['accuracy']
    )


    # 创建一个TensorBoard实例
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                                 histogram_freq=1,  # 每1个epoch记录一次
                                                 write_graph=True,  # 包含计算图
                                                 write_images=True)  # 可视化模型权重

    BEST_DEFAULT_n = {
        "clipnorm": 1,
        "optimizer": "Rmsprop",
        "batch_size": 64,
        "size": 128,
        "epochs": 20,
        "base_lr": 0.0001,
        "decay_lr": 0.9,
        "backbone_activation": "relu",
        "backbone_dr": 0.2,
        "forget_bias": 1.5,
        "backbone_units": 64,
        "backbone_layers": 1,
        "weight_decay": 1e-06,
        "use_mixed": False,
        "no_gate": False,
    }  # 预测多天的参数

    BEST_DEFAULT_n = json.dumps(BEST_DEFAULT_n, indent=4)
    tf.summary.text("model_params", tf.convert_to_tensor(BEST_DEFAULT_n))


    # 清理缓存
    callbacks = [
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: K.clear_session())
    ]

    start_time = time.time()
    # 训练模型
    history = model.fit(
        x=(x_train, x_date_train),
        y=y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=((x_valid, x_date_valid), y_valid),
        callbacks=[BackupCallback(model), callbacks, tensorboard],
        verbose=2,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_minutes = elapsed_time / 60.0
    print(f"Total time taken for training: {elapsed_time_minutes:.4f} minutes")

    # # 获取模型权重
    # weights = model.get_weights()
    #
    # # 遍历每一层并保存权重到CSV文件
    # for i in range(len(weights)):
    #     layer = model.layers[i]
    #     w = weights[i]
    #     layer_name = layer.name
    #     df = pd.DataFrame(w.reshape(-1, w.shape[-1]))
    #     df.to_csv(f'{layer_name}_weights.csv', index=False, header=False)
    #
    # # 可视化第一层权重
    # first_layer_name = model.layers[0].name
    # first_layer_weights = pd.read_csv(f'{first_layer_name}_weights.csv', header=None)
    #
    # # 根据权重矩阵的形状调整可视化代码
    # w = weights[0]
    # fig, axes = plt.subplots(nrows=w.shape[0], ncols=1, figsize=(10, 10 * w.shape[0]))
    # for i, ax in enumerate(axes.flat):
    #     ax.plot(first_layer_weights.iloc[i])
    #     ax.set_title(f'Neuron {i}')
    # plt.tight_layout()
    # plt.show()

    # 保存训练好的模型
    # model_path = 'Cfc_model.h5'
    # model.save(model_path, include_optimizer=False)



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

    # ——————————————————————训练过程——————————————————————————————————————————

    # 使用训练好的模型对测试数据进行预测
    predicted_values = model.predict((x_train, x_date_train))
    true_values = y_train  # y_train代表预测后相对应的标签，选取第一列特征即为收盘价


    predicted_values = predicted_values[:, n_days-1, None]  # 有n天的预测值，我需要选取预测的第几天的数据
    true_values = true_values[:, n_days-1, None]  # 有n天的预测值，我需要选取预测的第几天的数据

    # 反归一化
    predicted_values = predicted_values * (max_values - min_values) + min_values
    true_values = true_values * (max_values - min_values) + min_values

    # 提取第一列数据，即为收盘价
    predicted_values = predicted_values[:,0,None]
    true_values = true_values[:,0,None]

    # 计算模型的评价指标(标准化后)
    # R2 = r2_score(true_values, predicted_values)
    # MAE = mean_absolute_error(true_values, predicted_values)
    # RMSE = np.sqrt(mean_squared_error(true_values, predicted_values))
    # MAPE = np.mean(np.abs((true_values-predicted_values) / predicted_values))
    #
    # # 打印模型的评价指标
    # print('R2:', R2)
    # print('MAE:', MAE)
    # print('RMSE:', RMSE)
    # print('MAPE:', MAPE)

    # 画反归一化后日期的细致折线图
    # 转换为datetime对象
    # y_date_test = [datetime.fromtimestamp(ts) for ts in original_timestamps.flat]
    #
    # plt.figure(figsize=(12, 8))
    # plt.plot(y_date_test, predicted_values, label='预测值')
    # plt.plot(y_date_test, true_values, label='真实值')
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

    # 画平滑折线图
    # x_smooth = np.linspace(0, len(predicted_values), 100)  # 生成更多的数据点用于插值
    # spl_predicted = make_interp_spline(np.arange(len(predicted_values)), predicted_values.flatten(), k=2)
    # spl_true = make_interp_spline(np.arange(len(true_values)), true_values.flatten(), k=2)
    #
    # predicted_smooth = spl_predicted(x_smooth)
    # true_smooth = spl_true(x_smooth)
    #
    # plt.figure(figsize=(20, 16),dpi=400)
    # plt.plot(x_smooth, predicted_smooth, label='预测值', color='blue',alpha=0.8)
    # plt.plot(x_smooth, true_smooth, label='真实值', color='green',alpha=0.8)
    # plt.title('', fontsize=16)  # 设置标题字体大小
    # plt.xlabel('时间', fontsize=12)  # 设置x轴标签字体大小
    # plt.ylabel('股价', fontsize=12)  # 设置y轴标签字体大小
    # plt.xticks(fontsize=12)  # 设置x轴刻度字体大小
    # plt.yticks(fontsize=12)  # 设置y轴刻度字体大小
    # plt.legend()
    # plt.grid(True, alpha=0.5)
    # plt.tight_layout()
    # plt.show()

    # ---------------------end-------------------------

    # ————————————————————测试过程——————————————————————
    # custom_objects = {'CfcCell': tf_cfc.CfcCell}
    # model = tf.keras.models.load_model('Cfc_model.h5', custom_objects=custom_objects)

    # 测试集的结果
    predicted_test = model.predict((x_test, x_date_test))
    true_test = y_test

    predicted_test = predicted_test[:, n_days-1, None]  # 有n天的预测值，我需要选取预测的第几天的数据
    true_test = true_test[:, n_days-1, None]  # 有n天的预测值，我需要选取预测的第几天的数据

    # 反归一化
    predicted_test = predicted_test * (max_values - min_values) + min_values
    true_test = true_test * (max_values - min_values) + min_values

    predicted_test = predicted_test[:, 0, None]
    true_test = true_test[:, 0, None]

    # 计算模型的评价指标(标准化后)
    R2 = r2_score(true_test, predicted_test)
    MAE = mean_absolute_error(true_test, predicted_test)
    RMSE = np.sqrt(mean_squared_error(true_test, predicted_test))
    MAPE = np.mean(np.abs((true_test-predicted_test) / predicted_test))

    # 打印模型的评价指标
    print('R2:', R2)
    print('MAE:', MAE)
    print('RMSE:', RMSE)
    print('MAPE:', MAPE)

    #
    # # 画图
    # x_smooth = np.linspace(0, len(predicted_test), 100)  # 生成更多的数据点用于插值
    # spl_predicted = make_interp_spline(np.arange(len(predicted_test)), predicted_test.flatten(), k=2)
    # spl_true = make_interp_spline(np.arange(len(true_test)), true_test.flatten(), k=2)
    #
    # predicted_smooth = spl_predicted(x_smooth)
    # true_smooth = spl_true(x_smooth)
    #
    # plt.figure(figsize=(10, 8))
    # plt.plot(x_smooth, predicted_smooth, label='预测值', color='blue',alpha=0.8)
    # plt.plot(x_smooth, true_smooth, label='真实值', color='green',alpha=0.8)
    # plt.title('测试集结果')
    # plt.xlabel('时间')
    # plt.ylabel('股价')
    # plt.legend()
    # plt.grid(True, alpha=0.5)
    # plt.tight_layout()
    # plt.show()

    # 计算测试误差
    # test_errors = [(true - pred) / true for true, pred in zip(true_test, predicted_test)]

    # 绘制测试误差波动图
    # fig, ax = plt.subplots(figsize=(20, 6), dpi=200)
    # ax.axhline(y=0, color='r', linestyle='--', label='0误差线')  # 添加0误差线
    # ax.plot(range(1, len(test_errors) + 1), test_errors, marker='o', label='测试误差')
    # ax.set_title('', fontsize=12)
    # ax.set_xlabel('时间', fontsize=12)
    # ax.set_ylabel('测试误差', fontsize=12)
    # ax.tick_params(axis='x', labelsize=12)  # 设置x轴刻度字体大小
    # ax.tick_params(axis='y', labelsize=12)  # 设置y轴刻度字体大小
    # ax.legend(fontsize=12)
    # ax.grid(True, alpha=0.5)
    # plt.tight_layout()
    # plt.show()
    # ------------end--------------

    # 评估模型
    # test_loss = model.evaluate(
    #     x=(x_test, x_date_test),
    #     y=y_test,
    #     verbose=2,
    # )

    test_loss = model.evaluate(
        x=(x_test, x_date_test),
        y=y_test,
        verbose=2,
    )


    # ————————————————————————画测试误差图————————————————————————————
    # 计算测试误差
    test_errors = [(true - pred) / true for true, pred in zip(true_test, predicted_test)]
    # 绘制测试误差波动图
    fig, ax = plt.subplots(figsize=(12, 4), dpi=400)  # 增加dpi以获得更高清的图像
    ax.axhline(y=0, color='r', linestyle='--', label='')  # 添加0误差线

    # 使用细长柱状图绘制波动率信号图
    x = range(1, len(test_errors) + 1)
    y = test_errors
    for i in range(len(x)):
        ax.bar(x[i], y[i], color='blue', width=0.7, alpha=0.5)

    # 设置y轴的值域为正负
    ax.set_ylim([-0.15, 0.15])

    ax.set_title('', fontsize=24)
    ax.set_xlabel('时间', fontsize=20)
    ax.set_ylabel('测试误差', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)  # 设置x轴刻度字体大小
    ax.tick_params(axis='y', labelsize=16)  # 设置y轴刻度字体大小
    # ax.legend(fontsize=20,loc='upper right')
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig('test_error_signal.png', dpi=600, bbox_inches='tight')
    # -------------end----------------


    # 绘制三线合一图像————————————————————————————————————————
    # # 验证集的结果
    # predicted_valid = model.predict((x_valid, x_date_valid))
    # true_valid = y_valid
    #
    # predicted_valid = predicted_valid[:, n_days - 1, None]  # 有n天的预测值，我需要选取预测的第几天的数据
    # true_valid = true_valid[:, n_days - 1, None]  # 有n天的预测值，我需要选取预测的第几天的数据
    #
    # # 反归一化
    # predicted_valid = predicted_valid * (max_values - min_values) + min_values
    # true_valid = true_valid * (max_values - min_values) + min_values
    #
    # predicted_valid = predicted_valid[:, 0, None]
    # true_valid = true_valid[:, 0, None]
    #
    # # 训练集+验证集的结果
    # predicted_values_train_valid_test = np.concatenate((predicted_values, predicted_valid))
    # true_values_train_valid = np.concatenate((true_values, true_valid, true_test))
    #
    #
    # y_date_train = [datetime.fromtimestamp(ts) for ts in original_timestamps_train.flat]
    # y_date_valid = [datetime.fromtimestamp(ts) for ts in original_timestamps_valid.flat]
    # y_date_test = [datetime.fromtimestamp(ts) for ts in original_timestamps_test.flat]
    #
    # # 提取日期特征
    # date_train_valid_test = np.concatenate((y_date_train, y_date_valid, y_date_test))
    # date_train_valid = np.concatenate((y_date_train, y_date_valid))
    # date_test = y_date_test

    # ————————————————画后500个数据的图——————————————————
    # # 获取最后500个数据的索引
    # last_500_idx = range(-500, 0)
    # last_259_idx = range(-259, 0)
    # # 提取最后500个数据
    # date_train_valid_test_last_500 = np.array(date_train_valid_test)[last_500_idx]
    #
    # date_train_valid_last_500 = np.array(date_train_valid)[last_259_idx]
    #
    # true_values_train_valid_last_500 = true_values_train_valid[last_500_idx]
    #
    # predicted_values_train_valid_test_last_500 = predicted_values_train_valid_test[last_259_idx]
    #
    # # 计算10个平均分布的位置点
    # avg_indices1 = np.linspace(0, len(true_values_train_valid_last_500) - 1, 10, dtype=int)
    # avg_indices2 = np.linspace(0, len(predicted_values_train_valid_test_last_500) - 1, 6, dtype=int)
    # avg_indices3 = np.linspace(0, len(predicted_test) - 1, 4, dtype=int)
    #
    # avg_1 = [np.mean(predicted_values_train_valid_test_last_500[i:i + 50]) for i in avg_indices1]
    # avg_2 = [np.mean(true_values_train_valid_last_500[i:i + 50]) for i in avg_indices2]
    # avg_3 = [np.mean(true_values_train_valid_last_500[i:i + 50]) for i in avg_indices3]
    #
    # # 绘制折线图
    # fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    # ax.plot(date_train_valid_test_last_500, true_values_train_valid_last_500, color='#4A7298', label='真实值',
    #         linewidth=6)
    # ax.plot(date_train_valid_last_500, predicted_values_train_valid_test_last_500, color='#B7DBE3',
    #         label='训练集+验证集预测值', linestyle='--', linewidth=6)
    # ax.plot(date_test, predicted_test, label='测试集预测值', color='#F3C846', linestyle='-.',
    #         linewidth=6)
    # # ---------------------------end-----------------------
    #
    # # 画正常的三线合一————————————————————————————————————————
    # # fig, ax = plt.subplots(figsize=(16, 12), dpi=200)
    # # ax.plot(date_train_valid_test, true_values_train_valid, color='#4A7298', label='真实值',linewidth=3.5 )
    # # ax.plot(date_train_valid, predicted_values_train_valid_test, color='#B7DBE3', label='训练集+验证集的预测值', linestyle='--', linewidth=3.5 )
    # # ax.plot(date_test, predicted_test, label='测试集预测值',color='#F3C846', linestyle='-.', linewidth=3.5  )
    #
    # # 设置主刻度的格式
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #
    # # 设置次刻度
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    #
    # # 旋转x轴刻度
    # plt.xticks(rotation=45, ha='right', fontsize=24)
    # plt.yticks(fontsize=24)
    # plt.legend(loc='upper right', fontsize=26)
    # plt.ylabel('股价', fontsize=30)
    # plt.xlabel('时间', fontsize=30)
    # plt.title("", fontsize=24)
    # plt.tight_layout()
    # plt.grid(True, alpha=0.4)
    # # plt.show()
    # # 保存图片为高清PNG文件
    # plt.savefig('stock_prediction_new.png', dpi=600, bbox_inches='tight')
    # ------------------end---------------

    return test_loss, history
# -----------------------end--------------------------------------


# 默认参数设置
BEST_DEFAULT = {
    "clipnorm": 1,
    "optimizer": "Rmsprop",
    "batch_size": 64,
    "size": 128,
    "epochs": 20,
    "base_lr": 0.0001,
    "decay_lr": 0.9,
    "backbone_activation": "relu",
    "backbone_dr": 0,
    "forget_bias": 1.0,
    "backbone_units": 64,
    "backbone_layers": 2,
    "weight_decay": 3e-06,
    "use_mixed": False,
    "no_gate": False,
}
BEST_DEFAULT_n = {
    "clipnorm": 1,
    "optimizer": "Rmsprop",
    "batch_size": 64,
    "size": 128,
    "epochs": 50,
    "base_lr": 0.0001,
    "decay_lr": 0.98,
    "backbone_activation": "relu",
    "backbone_dr": 0.2,
    "forget_bias": 1.0,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-06,
    "use_mixed": False,
    "no_gate": True,
}  # 预测多天的参数

# MSE: 0.61654 +- 0.00634
BEST_MIXED = {
    "clipnorm": 1,
    "optimizer": "Rmsprop",
    "batch_size": 64,
    "size": 128,
    "epochs": 20,
    "base_lr": 0.0001,
    "decay_lr": 0.9,
    "backbone_activation": "relu",
    "backbone_dr": 0,
    "forget_bias": 1.2,
    "backbone_units": 128,
    "backbone_layers": 2,
    "weight_decay": 1e-05,
    "use_mixed": True,
    "no_gate": False,
}
# 0.65040 $\pm$ 0.00814
BEST_NO_GATE = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 96,
    "size": 256,
    "epochs": 50,
    "base_lr": 0.008,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.1,
    "forget_bias": 2.8,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 3e-05,
    "use_mixed": False,
    "no_gate": True,
}
# 0.94844 $\pm$ 0.00988
BEST_MINIMAL = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 1024,
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
    "batch_size": 1024,
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
    # 调用 eval 函数并使用提供的配置
    acc = eval(config, 1)
    print(f"Result: {acc}")  # 直接打印整个元组

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # score(BEST_DEFAULT)
    score(BEST_DEFAULT_n)
    # score(BEST_MIXED)
    # score(BEST_NO_GATE)
    # score(BEST_MINIMAL)
    # score(BEST_LTC)

    # # 在测试集上重复评估10次,计算指标平均值和标准差，稳健性实验
    # mae_scores = []
    # mape_scores = []
    # for i in range(10):
    #     y_pred = model.predict(x_test)
    #     mae = mean_absolute_error(y_test, y_pred)
    #     mape = mean_absolute_percentage_error(y_test, y_pred)
    #     mae_scores.append(mae)
    #     mape_scores.append(mape)
    #
    # print(f'MAE mean: {np.mean(mae_scores)}, MAE std: {np.std(mae_scores)}')
    # print(f'MAPE mean: {np.mean(mape_scores)}, MAPE std: {np.std(mape_scores)}')

# ---------------end-----------------------

# 生成五次，用五个模型
# def score(config):
   # losses = []
   #  histories = []
   #  for i in range(5):
   #      loss, history = eval(config, i)
   #      losses.append(loss)
   #      histories.append(history)
   #
   #  mean_loss = np.mean(losses)
   #  std_loss = np.std(losses)
   #  print(f"MSE: {mean_loss:0.5f} +/- {std_loss:0.5f}")



# 生成五次用五个模型
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--use_mixed", action="store_true")
#     parser.add_argument("--no_gate", action="store_true")
#     parser.add_argument("--minimal", action="store_true")
#     parser.add_argument("--use_ltc", action="store_true")
#
#     args = parser.parse_args()
#
#     if args.minimal:
#         score(BEST_MINIMAL)
#     elif args.no_gate:
#         score(BEST_NO_GATE)
#     elif args.use_ltc:
#         score(BEST_LTC)
#     elif args.use_mixed:
#         score(BEST_MIXED)
#     else:
#         score(BEST_DEFAULT)



