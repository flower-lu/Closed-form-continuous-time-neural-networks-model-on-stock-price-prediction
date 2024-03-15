from tkinter import *

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import tushare as ts 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def QandA():
 
    # ----参数配置--------------------
    distance_test = 0  # 训练数据的截止点离最新数据的距离
    # 输入：
    predict_number = gui.getPredict_date()  # 连续预测天数
    # 已经训练过的输入数据作为预测时的输入。由于LSTM隐含层和历史输入数据相关，
    # 当用于预测时，需要用一段训练数据作为预测输入，但该段数据
    # 的预测结果没有意义，仅仅是让模型隐含层记忆历史数据

    offset = 30
    # 训练数据的截止点离最近数据的真实距离，因为训练是以batch_size为单位进行训练的。
    # 因此real_distance_test大于等于distance_test.
    # ------end------------------

    # -------训练和预测过程--------------------
    instance_train_prediction = train_prediction(distance_test)
    instance_train_prediction.train_lstm()
    # instance_train_prediction.prediction(offset,predict_number)
    error_all = instance_train_prediction.prediction(offset, predict_number)
    print('error_all=', error_all)
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
    gui.draw(label, pre_predict, predict)
    print("finish")
    # ---------------end--------------------------


# ————————算法模型——————————
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_first):
        super(lstm_reg, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first)  # 隐藏层
        self.reg = nn.Linear(hidden_size, output_size)  # 输出层

    def forward(self, x):
        x, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        x = self.reg(x)
        return x


# -------end----------------------

class train_prediction():
    def __init__(self, distance_test):

        self.distance_test = distance_test

        input_size = 6  # 输入维度
        output_size = 6  # 输出维度
        num_layers = 1  # 层数
        # -----------------------------
        # 0--代表每次运算都随机初始化模型的权重参数，运行完以权重文件形式存储权重参数
        # 1（非零即可）--代表每次运算都使用上次运算的模型权重参数（上次运算权重已经存储在文件中）
        # 注意，若没有权重文件存在，而运行1，则程序会由于找不到权重文件而出错。
        switch_model = 0
        hidden_size = 10  # 隐含层神经元数目

        # 存储权重
        self.net1 = lstm_reg(input_size, hidden_size, output_size, num_layers, batch_first=True)
        if switch_model != 0:
            self.net1.load_state_dict(torch.load('./model_w.pkl'))

        ts.set_token(gui.getToken()) # 需要自己到网站www.tushare.pro注册获得token
        df = ts.pro_bar(ts_code=gui.getStock(), adj='qfq', start_date=gui.getStart_date(), end_date=gui.getEnd_date())

        time_step = 5  # 输入的时间步，即一次输入多少天数据
        
        # # -----获取数据--------------------------------
        df.to_csv('test.csv', index=False)
        data = np.array(df.loc[:, ['open', 'high', 'low', 'amount', 'vol', 'close']])

        # ----处理数据---------------------
        self.data = data[::-1]  # 反转，使数据按照日期先后顺序排列
        
        
        length = self.data.shape[1]
        for i in range(length):
            self.mean_data = np.mean(self.data[:, i])
            self.std_data = np.std(self.data[:, i])
            self.data[:, i] = (self.data[:, i] - self.mean_data) / self.std_data  # 标准化/归一化
        
        
        train_x, train_y = [], []  # 训练集
        for i in range(len(data) - time_step):
            x = self.data[i:i + time_step]
            y = self.data[i + 1:i + time_step + 1]
            train_x.append(x)
            train_y.append(y)

        self.train_x = torch.Tensor(train_x)
        self.train_y = torch.Tensor(train_y)
        self.var_x = Variable(self.train_x)
        self.var_y = Variable(self.train_y)
        
    # ———————————训练模型——————————
    def train_lstm(self):
        lr = gui.getLr()  # 学习率
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net1.parameters(), lr=lr)

        # 用户输入轮次
        epoch = gui.getEpoch()
        # 用户输入
        batch_size = int(gui.getBatch_size())  # 每一批次训练多少个样例
        
        for i in range(int(epoch)):
            pair = []
            for j in range(len(self.train_x) - self.distance_test):
                pair.append((self.var_x[j], self.var_y[j]))

            train_batches = DataLoader(dataset=pair, batch_size=batch_size, shuffle=True, pin_memory=True)
            for (x, y) in train_batches:

                out = self.net1(x)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            f = open('1.txt', mode='a')
            print('i=', i, '   loss=', loss.detach().numpy())
            f.write(str(i))
            f.write('  ')
            f.write(str(loss.detach().numpy()))
            f.write('\n')
            f.close()
        # --------------------------------------

        torch.save(self.net1.state_dict(), './model_w.pkl')

        # ----------end-------------------

    # ————————————用模型预测未来数据————————————
    def prediction(self, offset, predict_number):
        # 输入token
        # 以折线图展示data
        plt.figure()
        plt.plot(self.data[:, -1])
        # plt.show()
        
        
        
        # normalize_data=normalize_data[:,np.newaxis]       #增加维度x
        
        end2 = len(self.train_x) - self.distance_test - offset  # 扣除offset的起始点位置
        prev_seq = self.train_x[end2]  # 预测起始点的输入
        label = []  # 测试标签
        pre_predict = []  # 记录用训练数据预测的结果，数据无意义，仅用于隐含层记忆历史数据
        predict = []  # 有效的预测结果

        error_all = 0
        error_count = 0
        prev_seq = torch.Tensor(prev_seq)
        for i in range(offset + predict_number):
            prev_seq = torch.unsqueeze(prev_seq, 0)

            next_seq = self.net1(prev_seq)
            if i < self.distance_test + offset:
                label.append(self.train_y[end2 + i][-1][-1])
            if i < offset:  # 用训练集输入用于预测，预测结果无意义

                pre_predict.append(next_seq[-1][-1][-1].detach().numpy())
                # --------------------------------------

                prev_seq = self.train_y[end2 + i]
            else:  # 用上步预测结果作为当前步的输入，进行连续有效预测

                predict.append(next_seq[-1][-1][-1].detach().numpy())
                # --------------------------------------
                if i <= len(self.var_x) - end2 - 1:
                    error_label = self.train_y[end2 + i][-1][-1].numpy()
                    error_predict = next_seq[-1][-1][-1].detach().numpy()
                    error = abs(error_predict - error_label) / error_label
                    error_all = error_all + error
                    error_count = error_count + 1
                # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
                prev_seq = torch.squeeze(prev_seq, dim=0)
                prev_seq = torch.cat((prev_seq[1:], next_seq[:, -1]))

        label = np.array(label)
        error_all = error_all / (1 + error_count)
        predict = np.array(predict)
        pre_predict = np.array(pre_predict)
        # ---------end---------------

        # -------数据复原，反归一化-------------------------
        label = label * self.std_data + self.mean_data
        predict = predict * self.std_data + self.mean_data
        pre_predict = pre_predict * self.std_data + self.mean_data

        # ----------end-----------

        # -------数据输出--------------

        np.savez('./index.npz', label, pre_predict, predict)  # 保存数据，用于画图。可运行draw.py作图
        return error_all
    # --------end---------------------------


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
        Label(root,text='end_date: ', justify=RIGHT, width=50).place(x=10, y=250, width=80, height=20)
        Label(root, text='预测天数: ', justify=RIGHT, width=50).place(x=10, y=290, width=90, height=20)
        
        # 设置输出标签组件
        Label(root,text='五天预测值为： ', justify=RIGHT, width=50).place(x=10, y=400, width=100, height=20)
        # 创建输入文本框组件
        
        sVar = StringVar

        tokenValue = sVar(root, value='0c26f67a6b1eba09bab55f66ac6611b10c7a22018d859765fb18f3ac ')
        self.token = Entry(root, width=80, textvariable=tokenValue)
        self.token.place(x=100, y=10, width=600, height=20)
        
        self.epoch = Entry(root, width=80, textvariable=sVar(root, value='50'))
        self.epoch.place(x=100, y=50, width=80, height=20)
        
        self.batch_size = Entry(root, width=80, textvariable=sVar(root, value='30'))
        self.batch_size.place(x=100, y=90, width=80, height=20)
        
        self.lr = Entry(root, width=80, textvariable=sVar(root, value='0.004'))
        self.lr.place(x=100, y=130, width=80, height=20)
        
        self.stock = Entry(root, width=80, textvariable=sVar(root, value='000408.SZ'))
        self.stock.place(x=100, y=170, width=80, height=20)
        
        self.start_date = Entry(root, width=80, textvariable=sVar(root, value='20200220'))
        self.start_date.place(x=100, y=210, width=80, height=20)
        
        self.end_date = Entry(root, width=80, textvariable=sVar(root, value='20210220'))
        self.end_date.place(x=100, y=250, width=80, height=20)
        
        self.predict_date = Entry(root, width=80, textvariable=sVar(root, value='5'))
        self.predict_date.place(x=100, y=290, width=80, height=20)
        
        
        # 创建输出文本框组件
        self.prediction = Listbox(root, width=80)
        self.prediction.place(x=10, y=440, width=150, height=100)

        # 创建计算按钮组件，cal()主要把数据从输入文本框取出来
    

        Button(root,text='运行',command=QandA).place(x=30, y=350, width=70, height=30)
        
        fig = Figure(figsize=(6,4),dpi=150)#图像比例
        self.ax = fig.add_subplot(1, 1, 1)
        self.canvas_spice = FigureCanvasTkAgg(fig, root)
        self.canvas_spice.get_tk_widget().place(x=200,y=50)#放置位置
        root.protocol("WM_DELETE_WINDOW", self.closeWindow)

    def setPredict(self, p):
        i = 0
        self.prediction.delete(0, END)
        for s in p:
            self.prediction.insert(i, s)
            i = i + 1

    def draw(self, label, pre_predict, predict):
        self.ax.clear()
        self.ax.plot(list(range(len(label))), label, color='b')
        self.ax.plot(list(range(len(pre_predict))), pre_predict, color='r')
        self.ax.plot(list(range(len(pre_predict), len(pre_predict) + len(predict))), predict, color='y')
        self.ax.set_xlim(1, len(pre_predict) + len(predict))
        self.ax.set_ylim(min(min(label), min(pre_predict), min(predict)), max(max(label), max(pre_predict), max(predict)))
        self.canvas_spice.draw()

    def getToken(self):
        return self.token.get()

    def getEpoch(self):
        return int(self.epoch.get())

    def getBatch_size(self):
        return int(self.batch_size.get())

    def getLr(self):
        return float(self.lr.get())

    def getStock(self):
        return self.stock.get()

    def getStart_date(self):
        return self.start_date.get()

    def getEnd_date(self):
        return self.end_date.get()

    def getPredict_date(self):
        return int(self.predict_date.get())

    def closeWindow(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
  

if __name__ == '__main__':
    gui = GUI()
    mainloop()


