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
import itertools
from sklearn.model_selection import ParameterGrid
from tensorflow.keras import optimizers

import tensorflow.keras.backend as K

import json

# tf.config.optimizer.set_jit(True)
# 环境问题处理
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

# tf.config.optimizer.set_jit(False)
# os.environ.pop('TF_XLA_FLAGS', None)
# 图表中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")  # 忽略一些警告 不影响运行

from tensorflow.keras import mixed_precision

# 设置策略为混合精度
mixed_precision.set_global_policy('mixed_float16')


# 设置允许TensorFlow动态增长GPU内存使用，这可以避免在GPU显存不足时失败,但需要格外注意内存管理。
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ————————————————————————网格搜索调参————————————————————————————————
# 定义需要调整的参数及其取值范围
param_grid = {
    'int_sequence_len': [5],
    'n_days': [1], # 固定预测天数消融实验
    'step_size': [1],
    'clipnorm': [1,10],
    'optimizer': ['Rmsprop'],
    'batch_size': [64, 128, 256, 512],
    'size': [128, 192, 256],
    'epochs': [20],
    'base_lr': [0.0001, 0.0005, 0.001],
    'decay_lr': [0.9, 0.95, 0.98],
    'backbone_activation': ['relu', 'lecun'],
    'backbone_dr': [0.2],
    'forget_bias': [1.0, 1.5, 1.8],
    'backbone_units': [64, 128, 256],
    'backbone_layers': [1, 2, 3],
    'weight_decay': [1e-6, 3e-6, 1e-5],
    'use_mixed': [False],  # 将布尔值包装在列表中
    'no_gate': [False],    # 将布尔值包装在列表中
    'use_ltc': [False]     # 将布尔值包装在列表中
}

# 生成所有参数组合
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())] # 别搞太多参数以防显存不够



# 定义评估函数
def evaluate_model(params):
    # 使用指定的参数训练模型
    int_sequence_len = params['int_sequence_len']
    n_days = params['n_days']
    step_size = params['step_size']
    config = {
        'clipnorm': params['clipnorm'],
        'optimizer': params['optimizer'],
        'batch_size': params['batch_size'],
        'size': params['size'],
        'epochs': params['epochs'],
        'base_lr': params['base_lr'],
        'decay_lr': params['decay_lr'],
        'backbone_activation': params['backbone_activation'],
        'backbone_dr': params['backbone_dr'],
        'forget_bias': params['forget_bias'],
        'backbone_units': params['backbone_units'],
        'backbone_layers': params['backbone_layers'],
        'weight_decay': params['weight_decay'],
        'use_mixed': params['use_mixed'],
        'no_gate': params['no_gate'],
        'use_ltc': params['use_ltc'],
    }

    test_loss, history = eval(config, int_sequence_len, n_days, step_size, 1)
    # 返回评估指标作为评判依据
    return test_loss

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
processor = DataProcessor("maotai_factor1.csv")

data_pca = processor.get_data_pca()
data_pca=np.array(data_pca)

# data_fa = processor.get_data_fa()
# data_fa=np.array(data_fa)

data_x=data_pca


# ——————————————————非日期数据标准化——————————————————————
# minmax归一化
num_features = data_x.shape[1] # 获取特征数量
# 计算每个特征的最大值和最小值
min_values = data_x.min(axis=0) # 按列计算最小值
max_values = data_x.max(axis=0) # 按列计算最大值
# 进行归一化
data_x = (data_x - min_values) / (max_values - min_values)

# --------------------end--------------------------------------

from tensorflow.keras import optimizers
# 余弦退火学习率更新法
# class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
#     def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
#         super(CosineAnnealingLRScheduler, self).__init__()
#         self.total_step = total_step
#
#         if warmth_rate == 0:
#             self.warm_step = 1
#         else:
#             self.warm_step = int(self.total_step * warmth_rate)
#
#         self.lr_max = lr_max
#         self.lr_min = lr_min
#
#     @tf.function
#     def __call__(self, step):
#         if step < self.warm_step:
#             lr = self.lr_max / self.warm_step * step
#         else:
#             lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos((step - self.warm_step) / self.total_step * np.pi))
#
#         return lr


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

def eval(config, int_sequence_len, n_days, step_size, index_arg):
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

    # 手动选用单元模块
    cell = CfcCell(units=config["size"], hparams=config)
    # cell = MixedCfcCell(units=config["size"], hparams=config)
    # cell = LTCCell(units=config["size"], hparams=config)

    pixel_input = tf.keras.Input(shape=(int_sequence_len, int_a), name="input")
    time_input = tf.keras.Input(shape=(int_sequence_len, 1), name="time")

    # 用的是RNN架构
    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)

    output_units = n_days  # n天的维度
    output_layer = tf.keras.layers.Dense(output_units)
    output_states = rnn((pixel_input, time_input))
    y = output_layer(output_states)

    model = tf.keras.Model(inputs=[pixel_input, time_input], outputs=[y])
    # model.summary()

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

    # 清理缓存
    callbacks = [
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: K.clear_session())
    ]
    # start_time = time.time()
    # 训练模型
    history = model.fit(
        x=(x_train, x_date_train),
        y=y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=((x_valid, x_date_valid), y_valid),
        callbacks=[BackupCallback(model),callbacks],
        verbose=0,
    )


    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # elapsed_time_minutes = elapsed_time / 60.0
    # print(f"Total time taken for training: {elapsed_time_minutes:.4f} minutes")

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

    # #计算模型的评价指标(标准化后)
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
    # print('MAE:', MAE)
    # print('RMSE:', RMSE)
    # print('MAPE:', MAPE)


    # ——————————画测试误差图————————————————
    # # 计算测试误差
    # test_errors = [(true - pred) / true for true, pred in zip(true_test, predicted_test)]
    #
    # # 绘制测试误差波动图
    # fig, ax = plt.subplots(figsize=(20, 6), dpi=300)  # 增加dpi以获得更高清的图像
    # ax.axhline(y=0, color='r', linestyle='--', label='0误差线')  # 添加0误差线
    #
    # # 使用直线段绘制波动率信号图
    # x = range(1, len(test_errors) + 1)
    # y = test_errors
    # for i in range(len(x) - 1):
    #     ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color='b', linewidth=1.5)
    #
    # ax.set_title('测试误差波动图', fontsize=16)
    # ax.set_xlabel('时间', fontsize=14)
    # ax.set_ylabel('测试误差', fontsize=14)
    # ax.tick_params(axis='x', labelsize=12)  # 设置x轴刻度字体大小
    # ax.tick_params(axis='y', labelsize=12)  # 设置y轴刻度字体大小
    # ax.legend(fontsize=14)
    # ax.grid(True, alpha=0.5)
    # plt.tight_layout()
    #
    # # 保存图片为高清PNG文件
    # plt.savefig('test_error_signal.png', dpi=300, bbox_inches='tight')


    # ——————————————————————画三线合一图——————————————————————————
    # 验证集的结果
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
    # predicted_values_train_valid = np.concatenate(
    #     (predicted_values,predicted_valid )
    # true_values_train_valid = np.concatenate(true_values, true_valid)
    #
    # # 日期反归一化
    # original_timestamps_train = y_date_train * (timestamp_max_param - timestamp_min_param) + timestamp_min_param
    # original_timestamps_valid = y_date_valid * (timestamp_max_param - timestamp_min_param) + timestamp_min_param
    # original_timestamps_test = y_date_test * (timestamp_max_param - timestamp_min_param) + timestamp_min_param
    #
    # # 取整
    # original_timestamps_train = original_timestamps.astype(int)
    # original_timestamps_valid = original_timestamps.astype(int)
    # original_timestamps_test = original_timestamps.astype(int)
    #
    # y_date_train = [datetime.fromtimestamp(ts) for ts in original_timestamps.flat]
    # y_date_valid = [datetime.fromtimestamp(ts) for ts in original_timestamps.flat]
    # y_date_test = [datetime.fromtimestamp(ts) for ts in original_timestamps.flat]
    #
    # # 提取日期特征
    # date_train_valid = np.concatenate((y_date_train, y_date_valid))
    # date_test = y_date_test
    #
    # # 绘制图像
    # fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    # ax.plot(date_train_valid, true_values_train_valid, label='训练集+验证集真实值', marker='o', markersize=4)
    # ax.plot(date_train_valid, predicted_values_train_valid, label='训练集+验证集预测值', marker='o', markersize=4)
    # ax.plot(date_test, predicted_test, label='测试集预测值', marker='o', markersize=4)
    #
    # # 设置主刻度的格式
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #
    # # 设置次刻度
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    #
    # # 旋转x轴刻度
    # plt.xticks(rotation=45, ha='right', fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.legend(loc='best', fontsize=12)
    # plt.ylabel('股价', fontsize=14)
    # plt.xlabel('时间', fontsize=14)
    # plt.title("股价预测", fontsize=16)
    # plt.tight_layout()
    #
    # # 保存图片为高清PNG文件
    # plt.savefig('stock_prediction_new.png', dpi=300, bbox_inches='tight')


    # 得出测试loss
    test_loss = model.evaluate(
        x=(x_test, x_date_test),
        y=y_test,
        verbose=0,
    )

    return test_loss, history
# -----------------------end---------------------


# ——————————主程序运行模块——————————
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # score(BEST_DEFAULT)
    # score(BEST_MIXED)
    # score(BEST_NO_GATE)
    # score(BEST_MINIMAL)
    # score(BEST_LTC)

    # 网格搜索调参法遍历运行
    # 遍历所有参数组合,找出最优组合
    best_params = None
    best_score = float('inf')
    iteration = 0
    for params in all_params:
        iteration += 1
        score = evaluate_model(params)
        if score < best_score:
            best_params = params
            best_score = score
        print(f'Score: {score}')
        print(f'Iteration: {iteration}')
        print(f'Parameters: {params}')
        print('-' * 50)  # 打印长线条
        print()

    print(f'Best Parameters: {best_params}, Best Score: {best_score}')





