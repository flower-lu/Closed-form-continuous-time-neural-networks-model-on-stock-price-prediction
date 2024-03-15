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
from tkinter import *
from matplotlib.figure import Figure
import tushare as ts
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# 环境问题处理
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# 图表中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")  # 忽略一些警告 不影响运行


def config():
    clipnorm = gui.getclipnorm()
    optimizer = gui.getoptimizer()
    batch_size = gui.getbatch_size()
    size = gui.getsize()
    epochs = gui.getepochs()
    base_lr = gui.getbase_lr()
    decay_lr = gui.getdecay_lr()
    backbone_activation = gui.getbackbone_activation()
    backbone_dr = gui.getbackbone_dr()
    forget_bias = gui.getforget_bias()
    backbone_units = gui.getbackbone_units()
    backbone_layers = gui.getbackbone_layers()
    weight_decay = gui.getweight_decay()
    use_mixed = gui.getuse_mixed()
    no_gate = gui.getno_gate()
    minimal = gui.getminimal()
    use_ltc = gui.getuse_ltc()
    return clipnorm,optimizer,batch_size,size,epochs,base_lr,decay_lr,backbone_activation,backbone_dr,forget_bias,backbone_units,backbone_layers,weight_decay,use_mixed,no_gate,minimal,use_ltc

def QandA():
    # ----参数配置--------------------
    distance_test = 0  # 训练数据的截止点离最新数据的距离

    offset = 30
    # 训练数据的截止点离最近数据的真实距离，因为训练是以batch_size为单位进行训练的。
    # ------end------------------

    # -------训练和预测过程--------------------
    # instance_train_prediction = train_prediction(distance_test)
    # instance_train_prediction.train_lstm()
    # error_all = instance_train_prediction.prediction(offset, predict_number)
    # print('error_all=', error_all)
    # ----------------end-----------------------

    # ---------数据可视化表示----------------
    D = np.load('./index.npz')
    label = D['arr_0']
    pre_predict = D['arr_1']
    predict = D['arr_2']
    D.close()
    print('label=', label)
    print('predict=', predict)

    # return predict
    gui.setPredict(predict)
    gui.draw(label, predict)
    print("finish")


    ts.set_token(gui.getToken())  # 需要自己到网站www.tushare.pro注册获得token
    df = ts.pro_bar(ts_code=gui.getStock(), adj='qfq', start_date=gui.getStart_date(), end_date=gui.getEnd_date())

    df.to_csv('test.csv', index=False)
    data_date = np.array(df.loc[:, ['date']])
    data = np.array(df.loc[:, ['open', 'high', 'low', 'amount', 'vol', 'close']])

    # ----处理数据---------------------
    data = data[::-1]  # 反转，使数据按照日期先后顺序排列
    data_date = data_date[::-1]

    # minmax归一化
    num_features = data.shape[1] # 获取特征数量
    # 计算每个特征的最大值和最小值
    min_values = data.min(axis=0) # 按列计算最小值
    max_values = data.max(axis=0) # 按列计算最大值
    # 进行归一化
    data_x = (data - min_values) / (max_values - min_values)


    # ——————————————特征处理————————————————————————————————————————————————
    # 将日期列转换为日期时间类型
    data_date = pd.to_datetime(data_date, format='%Y%m%d')
    # 将日期时间转换为时间戳（以秒为单位）
    data['timestamp'] = data_date.apply(lambda x: x.timestamp()).astype(int)
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


    # 调用文件，用PCA、PCA获取降维后的特征
    # processor = DataProcessor("shangzheng_factor.csv")

    # data_pca = processor.get_data_pca()
    # data_pca=np.array(data_pca)

    # data_fa = processor.get_data_fa()
    # data_fa=np.array(data_fa)

    # data_x=data_pca
    # data_x=data_fa
    # ----------------------end--------------------------------------

    # ——————————————————————————样本数据标签数据处理—————————————————————————————————————
    # 移动时间窗口参数，int_sequence_len为输入天数，n_days为输出天数
    int_sequence_len = gui.getInput_date()  # 一次输入多少天数据，即输入的时间步，可以调整，为序列长度
    n_days = gui.getPredict_date  # 设置要预测的天数
    step_size = gui.getStep  # 滑动窗口的步长

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


    # 日期反归一化
    original_timestamps = y_date_train * (timestamp_max_param - timestamp_min_param) + timestamp_min_param
    # 取整
    original_timestamps = original_timestamps.astype(int)

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
    # 保存训练好的模型
    # model_path = 'Cfc_model.h5'
    # model.save(model_path, include_optimizer=False)

    # 记录训练损失
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    # 创建迭代数量
    epoch_count = range(1, len(training_loss) + 1)
    # 可视化损失历史
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.title("Epoch-loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
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

    lable = true_values
    predict = predicted_values
    # -------数据输出--------------

    np.savez('./index.npz', lable, predict)  # 保存数据，用于画图。可运行draw.py作图

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


    # 画训练集的图
    x_smooth = np.linspace(0, len(predicted_values), 100)  # 生成更多的数据点用于插值
    spl_predicted = make_interp_spline(np.arange(len(predicted_values)), predicted_values.flatten(), k=2)
    spl_true = make_interp_spline(np.arange(len(true_values)), true_values.flatten(), k=2)

    predicted_smooth = spl_predicted(x_smooth)
    true_smooth = spl_true(x_smooth)

    plt.figure(figsize=(10, 8))
    plt.plot(x_smooth, predicted_smooth, label='预测值', color='blue',alpha=0.8)
    plt.plot(x_smooth, true_smooth, label='真实值', color='green',alpha=0.8)
    plt.title('测试集结果')
    plt.xlabel('时间')
    plt.ylabel('股价')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()

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

    predicted_test = predicted_test[:,0,None]
    true_test = true_test[:,0,None]

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

    # 画测试集的图
    x_smooth = np.linspace(0, len(predicted_test), 100)  # 生成更多的数据点用于插值
    spl_predicted = make_interp_spline(np.arange(len(predicted_test)), predicted_test.flatten(), k=2)
    spl_true = make_interp_spline(np.arange(len(true_test)), true_test.flatten(), k=2)

    predicted_smooth = spl_predicted(x_smooth)
    true_smooth = spl_true(x_smooth)

    plt.figure(figsize=(10, 8))
    plt.plot(x_smooth, predicted_smooth, label='预测值', color='blue',alpha=0.8)
    plt.plot(x_smooth, true_smooth, label='真实值', color='green',alpha=0.8)
    plt.title('测试集结果')
    plt.xlabel('时间')
    plt.ylabel('股价')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


    # 计算测试误差
    test_errors = [(true - pred) / true for true, pred in zip(true_test, predicted_test)]
    # 绘制测试误差波动图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(y=0, color='r', linestyle='--', label='0误差线')  # 添加0误差线
    ax.plot(range(1, len(test_errors) + 1), test_errors, marker='o', label='测试误差')
    ax.set_title('测试误差波动图', fontsize=16)
    ax.set_xlabel('时间', fontsize=14)
    ax.set_ylabel('测试误差', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
    # ------------------end------------------

    # 评估模型
    # test_loss = model.evaluate(
    #     x=(x_test, x_date_test),
    #     y=y_test,
    #     verbose=2,
    # )

    test_loss = model.evaluate(
        x=(x_valid, x_date_valid),
        y=y_valid,
        verbose=2,
    )

    return test_loss
# -----------------------end--------------------------------------


# 综合评估结果
def score(config):
    # 调用 eval 函数并使用提供的配置
    acc = eval(config, 1)
    print(f"MSE: {acc:0.5f}")  # 直接打印MSE值


# 按钮控件
# command 按钮点击触发的事件
class GUI:
    def __init__(self):
        # 创建主窗口
        self.root = root = Tk()
        # 设置窗口大小
        root.geometry("1200x700")
        root.title("股票预测")
        # 设置输入标签组件
        Label(root, text='token: ', justify=RIGHT, width=50).place(x=10, y=10, width=80, height=20)
        Label(root, text='epoch: ', justify=RIGHT, width=50).place(x=10, y=50, width=80, height=20)
        Label(root, text='batch_size: ', justify=RIGHT, width=50).place(x=10, y=90, width=80, height=20)
        Label(root, text='lr: ', justify=RIGHT, width=50).place(x=10, y=130, width=80, height=20)
        Label(root, text='股票代码: ', justify=RIGHT, width=50).place(x=10, y=170, width=80, height=20)
        Label(root, text='start_date: ', justify=RIGHT, width=50).place(x=10, y=210, width=80, height=20)
        Label(root, text='end_date: ', justify=RIGHT, width=50).place(x=10, y=250, width=80, height=20)
        Label(root, text='预测天数: ', justify=RIGHT, width=50).place(x=10, y=290, width=90, height=20)

        # 设置输出标签组件
        Label(root, text='五天预测值为： ', justify=RIGHT, width=50).place(x=10, y=400, width=100, height=20)
        # 创建输入文本框组件

        sVar = StringVar

        tokenValue = sVar(root, value='0c26f67a6b1eba09bab55f66ac6611b10c7a22018d859765fb18f3ac ')
        self.token = Entry(root, width=80, textvariable=tokenValue)
        self.token.place(x=100, y=10, width=600, height=20)
        self.stock = Entry(root, width=80, textvariable=sVar(root, value='000408.SZ'))
        self.stock.place(x=100, y=170, width=80, height=20)
        self.start_date = Entry(root, width=80, textvariable=sVar(root, value='20200220'))
        self.start_date.place(x=100, y=210, width=80, height=20)
        self.end_date = Entry(root, width=80, textvariable=sVar(root, value='20210220'))
        self.end_date.place(x=100, y=250, width=80, height=20)
        self.n_days = Entry(root, width=80, textvariable=sVar(root, value='5'))
        self.n_days.place(x=100, y=290, width=80, height=20)
        self.int_sequence_len = Entry(root, width=80, textvariable=sVar(root, value='10'))
        self.int_sequence_len.place(x=100, y=330, width=80, height=20)
        self.step_size = Entry(root, width=80, textvariable=sVar(root, value='1'))
        self.step_size.place(x=100, y=370, width=80, height=20)

        self.clipnorm = Entry(root, textvariable=sVar(root, value='1.0'))
        self.optimizer = Entry(root, textvariable=sVar(root, value='1.0'))
        self.batch_size = Entry(root, textvariable=sVar(root, value='64'))
        self.size = Entry(root, textvariable=sVar(root, value='64'))
        self.epochs = Entry(root, textvariable=sVar(root, value='100'))
        self.base_lr = Entry(root, textvariable=sVar(root, value='0.001'))
        self.decay_lr = Entry(root, textvariable=sVar(root, value='0.0'))
        self.backbone_activation = Entry(root, textvariable=sVar(root, value='0.1'))
        self.backbone_dr = Entry(root, textvariable=sVar(root, value='0.0'))
        self.forget_bias = Entry(root, textvariable=sVar(root, value='1.0'))
        self.backbone_units = Entry(root, textvariable=sVar(root, value='128'))
        self.backbone_layers = Entry(root, textvariable=sVar(root, value='2'))
        self.weight_decay = Entry(root, textvariable=sVar(root, value='0'))
        self.use_mixed = Checkbutton(root, variable=BooleanVar())
        self.no_gate = Entry(root, textvariable=sVar(root, value='0'))
        self.minimal = Entry(root, textvariable=sVar(root, value='0'))
        self.use_ltc = Checkbutton(root, variable=BooleanVar())


        # 创建输出文本框组件
        self.prediction = Listbox(root, width=80)
        self.prediction.place(x=10, y=440, width=150, height=100)

        # 创建计算按钮组件，cal()主要把数据从输入文本框取出来

        Button(root, text='运行', command=QandA).place(x=30, y=350, width=70, height=30)

        fig = Figure(figsize=(6, 4), dpi=150)  # 图像比例
        self.ax = fig.add_subplot(1, 1, 1)
        self.canvas_spice = FigureCanvasTkAgg(fig, root)
        self.canvas_spice.get_tk_widget().place(x=200, y=50)  # 放置位置
        root.protocol("WM_DELETE_WINDOW", self.closeWindow)

    def setPredict(self, p):
        i = 0
        self.prediction.delete(0, END)
        for s in p:
            self.prediction.insert(i, s)
            i = i + 1

    def draw(self, label,  predict):
        self.ax.clear()
        self.ax.plot(list(range(len(label))), label, color='b')
        self.ax.plot(list(range(len(predict), len(predict))), predict, color='y')
        self.ax.set_xlim(1, len(predict))
        self.ax.set_ylim(min(min(label),  min(predict)),
                         max(max(label),  max(predict)))
        self.canvas_spice.draw()

    def getclipnorm(self):
        return float(self.clipnorm.get())
    def getoptimizer(self):
        return float(self.optimizer.get())
    def getbatch_size(self):
        return float(self.batch_size.get())
    def getsize(self):
        return float(self.size.get())
    def getepochs(self):
        return float(self.epochs.get())
    def getbase_lr(self):
        return float(self.base_lr.get())
    def getdecay_lr(self):
        return float(self.decay_lr.get())
    def getbackbone_activation(self):
        return float(self.backbone_activation.get())
    def getbackbone_dr(self):
        return float(self.backbone_dr.get())
    def getforget_bias(self):
        return float(self.forget_bias.get())
    def getbackbone_units(self):
        return self.backbone_units.get()
    def getbackbone_layers(self):
        return int(self.backbone_layers.get())
    def getweight_decay(self):
        return int(self.weight_decay.get())
    def getuse_mixed(self):
        return self.use_mixed.get()
    def getno_gate(self):
        return int(self.no_gate.get())
    def getminimal(self):
        return int(self.minimal.get())
    def getuse_ltc(self):
        return self.use_ltc.get()
    def getToken(self):
        return self.token.get()
    def getStock(self):
        return self.stock.get()
    def getStart_date(self):
        return self.start_date.get()
    def getEnd_date(self):
        return self.end_date.get()
    def getPredict_date(self):
        return int(self.n_days.get())
    def getInput_date(self):
        return int(self.int_sequence_len.get())
    def getStep(self):
        return int(self.step_size.get())
    def closeWindow(self):
        self.root.quit()  # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # score(BEST_DEFAULT)
    score(BEST_MIXED)
    # score(BEST_NO_GATE)
    # score(BEST_MINIMAL)
    # score(BEST_LTC)
    gui = GUI()
    mainloop()





