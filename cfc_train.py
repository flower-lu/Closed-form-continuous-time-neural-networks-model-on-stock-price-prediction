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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import binarize
from Factor_ans_use import DataProcessor
import tensorflow as tf
import argparse
from tf_cfc import CfcCell, MixedCfcCell, LTCCell
import time
import sys
# 调用模型评价指标
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# 图表中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")  # 忽略一些警告 不影响运行

# 读取数据
data = pd.read_csv("maotai_factor.csv",encoding='gbk')
data = data.fillna(0)

# 数据表转换为浮点数
data['close'] = data['close'].str.replace(',', '').astype(float)
data['open'] = data['open'].str.replace(',', '').astype(float)
data['high'] = data['high'].str.replace(',', '').astype(float)
data['low'] = data['low'].str.replace(',', '').astype(float)
# Convert columns to float without using .str accessor
data['amt'] = data['amt'].astype(float)
# Similarly, convert other columns to float without using .str accessor
data['close'] = data['close'].astype(float)
data['open'] = data['open'].astype(float)
data['high'] = data['high'].astype(float)
data['low'] = data['low'].astype(float)
data['MA5'] = data['MA5'].astype(float)
data['MA10'] = data['MA10'].astype(float)
data['MA20'] = data['MA20'].astype(float)
data['DIF'] = data['DIF'].astype(float)
data['DEA'] = data['DEA'].astype(float)
data['MACD'] = data['MACD'].astype(float)
data['K'] = data['K'].astype(float)
data['D'] = data['D'].astype(float)
data['J'] = data['J'].astype(float)
data['RSI'] = data['RSI'].astype(float)
data['WR'] = data['WR'].astype(float)
data['WR1'] = data['WR1'].astype(float)
data['BIAS1'] = data['BIAS1'].astype(float)
data['BIAS2'] = data['BIAS2'].astype(float)
data['BIAS3'] = data['BIAS3'].astype(float)
data['PSY'] = data['PSY'].astype(float)
data['PSYMA'] = data['PSYMA'].astype(float)
data['CCI'] = data['CCI'].astype(float)
data['ATR'] = data['ATR'].astype(float)
data['BBI'] = data['BBI'].astype(float)

# 特征选择
# 日期特征
data_date=data[['date']]
data_date=np.array(data_date)
# 全25特征
# data_x=data[['close', 'open', 'high', 'low','amt', 'MA5','MA10',
#                  'MA20','DIF','DEA','MACD','K','D','J','RSI','WR','WR1',
#                  'BIAS1','BIAS2','BIAS3','PSY','PSYMA','CCI','ATR','BBI',
#                  ]].values
# 关系最强的四特征
data_x=data[['close','open', 'high', 'low']].values
data_x=np.array(data_x)

#特征降维
# processor = DataProcessor("maotai_factor.csv")
# data_pca = processor.get_data_pca()
# data_pca=np.array(data_pca)
#
# data_kpca = processor.get_data_kpca()
# data_kpca=np.array(data_kpca)
#
# data_fa = processor.get_data_fa()
# data_fa=np.array(data_fa)
#
# data_x=data_fa

# print(data_x)
# print(data_date)

# 生成标签
data['label'] = np.where(data['close'].shift(-1) >= data['close'], 1, 0)

# 序列长度，输入的时间步，即一次输入多少天数据。可以调整
int_sequence_len=10
# 每个序列的长度，每个时间步数据的特征数量，即每个时间步的数据维度。固定的
int_a = data_x.shape[1]

train_x, train_y = [], []  # 训练集
for i in range(0, len(data_x)-int_sequence_len,1):
    train_x.append(data_x[i:i+int_sequence_len])
    train_y.append(data_x[i + int_sequence_len - 1])  # 注意这里是取前一天的标签

# train_x, train_y = [], []  # 训练集
# for i in range(len(data) - time_step):
#     x = self.data[i:i + time_step]
#     y = self.data[i + 1:i + time_step + 1]
#     train_x.append(x)
#     train_y.append(y)

# 构建训练集
train_date_x = []
train_date_y = []
for i in range(0, len(data_date)-int_sequence_len,1):
    train_date_x.append(data_date[i:i+int_sequence_len])
    train_date_y.append(data_date[i+int_sequence_len])


# 转换为数组，划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(np.array(train_x), np.array(train_y), test_size=0.2, random_state=1)
x_date_train, x_date_test, y_date_train, y_date_test = train_test_split(np.array(train_date_x),np.array(train_date_y), test_size=0.2, random_state=1)



# 进行数据归一化，将数据归一化到0-1之间
x_train = x_train.reshape(len(x_train), int_sequence_len * int_a)
x_test = x_test.reshape(len(x_test), int_sequence_len * int_a)

# 应用 MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 将 x_train 和 x_test 重塑回原来的形状
x_train = x_train.reshape(len(x_train), int_sequence_len, int_a)
x_test = x_test.reshape(len(x_test), int_sequence_len, int_a)


y_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_normalized = y_scaler.fit_transform(y_train)
y_test_normalized = y_scaler.transform(y_test)

# Reshape y_train and y_test back to their original shapes
y_train_normalized = y_train_normalized.reshape(len(y_train_normalized), -1)
y_test_normalized = y_test_normalized.reshape(len(y_test_normalized), -1)
y_train = y_train_normalized
y_test = y_test_normalized

date_scaler = MinMaxScaler(feature_range=(0, 1))
x_date_train_normalized = date_scaler.fit_transform(np.array(x_date_train).reshape(-1, 1))
x_date_test_normalized = date_scaler.transform(np.array(x_date_test).reshape(-1, 1))

y_date_train_normalized = date_scaler.fit_transform(np.array(y_date_train).reshape(-1, 1))
y_date_test_normalized = date_scaler.transform(np.array(y_date_test).reshape(-1, 1))

# Reshape back to original shape
x_date_train_resc = x_date_train_normalized.reshape(x_date_train.shape)
x_date_test_resc = x_date_test_normalized.reshape(x_date_test.shape)
x_date_train = x_date_train_resc
x_date_test = x_date_test_resc

y_date_train_resc = y_date_train_normalized.reshape(y_date_train.shape)
y_date_test_resc = y_date_test_normalized.reshape(y_date_test.shape)
y_date_train = y_date_train_resc
y_date_test = y_date_test_resc


# data_date = pd.DataFrame(data_date, columns=['date'])
# data_date['date'] = pd.to_datetime(data_date['date'])
# data_date['timestamp'] = data_date['date'].astype(np.int64) // 10**9
#
#
# date_scaler = MinMaxScaler(feature_range=(0, 1))
# data_date['timestamp'] = date_scaler.fit_transform(data_date[['timestamp']])
#
# print(data_date)

# print(x_train.shape)
# print(len(x_train), len(x_test))  # 1243 311


x_train = x_train.reshape(len(x_train),int_sequence_len, int_a) # 三维度数据 全部数据长度 序列长度 每个序列维度
y_train = y_train.reshape(len(x_train),int_a)
x_train, y_train = np.array(x_train), np.array(y_train)

# print(x_train.shape)
# print(y_train.shape)
x_test = x_test.reshape(len(x_test),int_sequence_len, int_a)
y_test = y_test.reshape(len(x_test),int_a)

x_test, y_test = np.array(x_test), np.array(y_test)



# 构建模型
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

    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
    dense_layer = tf.keras.layers.Dense(1)

    output_states = rnn((pixel_input, time_input))
    y = dense_layer(output_states)

    model = tf.keras.Model(inputs=[pixel_input, time_input], outputs=[y])

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
#
    # 训练模型
    history = model.fit(
        x=(x_train, x_date_train),
        y=y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=((x_test, x_date_test), y_test),
        callbacks=[BackupCallback(model)],
        verbose=2,
    )


    # 使用训练好的模型对测试数据进行预测
    predicted_values = model.predict((x_test, x_date_test))
    true_values = y_test[:, 0, None]

    # # 创建MinMaxScaler对象并使用原始数据集的所有特征
    # x_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    # x_scaler.fit(data['close'].values.reshape(-1, 1))
    #
    # # 将预测值和真实值进行反归一化
    # predicted_values_normalized = x_scaler.inverse_transform(predicted_values)
    # true_values_normalized = x_scaler.inverse_transform(true_values)
    #
    # # 将重塑后的预测值和真实值展平为一维数组
    # predicted_values_flat = predicted_values_normalized.flatten()
    # true_values_flat = true_values_normalized.flatten()
    #
    # predicted_values = predicted_values_flat.reshape(-1,1)
    # true_values = true_values_flat.reshape(-1,1)
    #
    #
    # print(predicted_values.shape)
    # print(true_values.shape)


    # 绘制图表
    plt.figure(figsize=(20, 12))
    plt.plot(predicted_values, label='预测值', color='blue')
    plt.plot(true_values, label='真实值', color='green')
    plt.title('股价预测结果对比')
    plt.xlabel('样本序号')
    plt.ylabel('股价')
    plt.legend()
    plt.show()

    # 计算模型的评价指标
    R2 = r2_score(true_values, predicted_values)
    MAE = mean_absolute_error(true_values, predicted_values)
    RMSE = np.sqrt(mean_squared_error(true_values, predicted_values))
    MAPE = np.mean(np.abs((true_values-predicted_values) / predicted_values))

    # 打印模型的评价指标
    print('R2:', R2)
    print('MAE:', MAE)
    print('RMSE:', RMSE)
    print('MAPE:', MAPE)


    # # 定义一个函数，用于将概率转换为类别（针对每个输出）
    # def convert_to_classes(probabilities):
    #     return (probabilities > 0.5).astype(int)
    #
    # # 对测试数据进行预测
    # predicted_probabilities = predicted_values
    #
    # # # 将概率转换为类别（针对每个输出）
    # # predicted_classes = [convert_to_classes(pred) for pred in predicted_probabilities]
    #
    # # 对测试数据进行预测
    # predicted_classes = convert_to_classes(predicted_probabilities)
    #
    # # 对测试数据进行预测
    # real_probabilities = true_values
    #
    # # 对测试数据进行预测
    # real_binary = convert_to_classes(real_probabilities)
    #
    # # real_binary = binarize(true_values, threshold=0.5)
    # # 计算每个输出的评估指标
    #
    # # 初始化评估指标和混淆矩阵
    # total_accuracy = 0
    # total_precision = 0
    # total_recall = 0
    # total_f1 = 0
    # total_conf_matrix = np.zeros((2, 2))
    #
    # # 遍历每个输出，计算评估指标和混淆矩阵
    # for true, pred in zip(real_binary, predicted_classes):
    #     total_accuracy += accuracy_score(true, pred)
    #     total_precision += precision_score(true, pred)
    #     total_recall += recall_score(true, pred)
    #     total_f1 += f1_score(true, pred)
    #     total_conf_matrix += confusion_matrix(true, pred)
    #
    # # Calculate average evaluation metrics
    # num_outputs = len(real_binary)
    # avg_accuracy = total_accuracy / num_outputs
    # avg_precision = total_precision / num_outputs
    # avg_recall = total_recall / num_outputs
    # avg_f1 = total_f1 / num_outputs
    #
    # # Calculate average confusion matrix
    # avg_conf_matrix = total_conf_matrix / num_outputs
    #
    # # Display results
    # print("平均准确率:", avg_accuracy)
    # print("平均精确率:", avg_precision)
    # print("平均召回率:", avg_recall)
    # print("平均F1 分数:", avg_f1)
    # print("平均混淆矩阵:")
    # print(avg_conf_matrix)

    # 评估模型
    test_loss = model.evaluate(
        x=(x_test, x_date_test),
        y=y_test,
        verbose=2,
    )

    return test_loss, plt , model

    # 使用训练好的模型对测试数据进行预测

# config = {
#     "use_ltc": False,  # 设置为 True 如果要使用 LTCCell
#     "use_mixed": False,  # 设置为 True 如果要使用 MixedCfcCell
#     "clipnorm": 10,  # 梯度裁剪
#     "optimizer": "rmsprop",
#     "batch_size": 128,
#     "size": 64,
#     "epochs": 200,
#     "base_lr": 0.005,
#     "decay_lr": 0.95,
#     "backbone_activation": "relu",
#     "backbone_dr": 0.0,
#     "forget_bias": 0.6,
#     "backbone_units": 128,
#     "backbone_layers": 1,
#     "weight_decay": 2e-06,
#     # 其他配置参数...
# }
# #
# best_test_acc, trained_model = eval(config, x_train, y_train, x_test, y_test, verbose=0)
# #
# print("最佳测试准确率:", best_test_acc)
#
# # 保存训练好的模型
# trained_model.save('CFC_stock_model.h5')
#


# Accuracy: 99.42% +- 0.42
# DENSE: 97.34\% $\pm$ 1.85
BEST_DEFAULT = {
    "clipnorm": 1,
    "optimizer": "rmsprop",
    "batch_size": 1024,
    "size": 192,
    "epochs": 50,
    "base_lr": 0.005,
    "decay_lr": 0.95,
    "backbone_activation": "relu",
    "backbone_dr": 0.0,
    "forget_bias": 1.2,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 3e-06,
    "use_mixed": False,
}

# MSE: 0.61654 +- 0.00634
BEST_MIXED = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 1024,
    "size": 128,
    "epochs": 50,
    "base_lr": 0.005,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.2,
    "forget_bias": 2.1,
    "backbone_units": 128,
    "backbone_layers": 2,
    "weight_decay": 6e-06,
    "use_mixed": True,
    "no_gate": False,
}

# 0.65040 $\pm$ 0.00814
BEST_NO_GATE = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 1024,
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

def score(config):
    acc = [eval(config, i) for i in range(5)]
    mse_values = [result[0] for result in acc]  # 提取模型评估结果中的 MSE 值
    print(f"MSE: {np.mean(mse_values):0.5f} $\\pm$ {np.std(mse_values):0.5f}")  # 计算并打印 MSE 的均值和标准差


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        score(LTC_TEST)
    elif args.use_mixed:
        score(BEST_MIXED)
    else:
        score(BEST_DEFAULT)


