import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import tushare as ts
import matplotlib.pyplot as plt

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

class train_prediction():
    def __init__(self, distance_test):
        self.distance_test = distance_test

        input_size = 6  # 输入维度
        output_size = 6  # 输出维度
        num_layers = 1  # 层数
        switch_model = 0
        hidden_size = 10  # 隐含层神经元数目

        # 存储权重
        self.net1 = lstm_reg(input_size, hidden_size, output_size, num_layers, batch_first=True)
        if switch_model != 0:
            self.net1.load_state_dict(torch.load('./model_w.pkl'))

        ts.set_token("0c26f67a6b1eba09bab55f66ac6611b10c7a22018d859765fb18f3ac")  # 需要自己到网站www.tushare.pro注册获得token
        df = ts.pro_bar(ts_code='000408.SZ', adj='qfq', start_date='20200220', end_date='20210220')

        time_step = 5  # 输入的时间步，即一次输入多少天数据

        data = np.array(df.loc[:, ['open', 'high', 'low', 'amount', 'vol', 'close']])

        self.data = data[::-1]  # 反转，使数据按照日期先后顺序排列

        length = self.data.shape[1]
        for i in range(length):
            self.mean_data = np.mean(self.data[:, i])
            self.std_data = np.std(self.data[:, i])
            self.data[:, i] = (self.data[:, i] - self.mean_data) / self.std_data  # 标准化

        train_x, train_y = [], []  # 训练集
        for i in range(len(data) - time_step):
            x = self.data[i:i + time_step]
            y = self.data[i + 1:i + time_step + 1]
            train_x.append(x)
            train_y.append(y)

        self.train_x = torch.Tensor(train_x)
        self.train_y = torch.Tensor(train_y)
        self.train_x = Variable(self.train_x)
        self.var_x = Variable(self.train_x)
        self.var_y = Variable(self.train_y)

    # ———————————训练模型——————————
    def train_lstm(self):
        lr = 0.004  # 学习率
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net1.parameters(), lr=lr)

        epoch = 50
        batch_size = 30  # 每一批次训练多少个样例

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

            print('Epoch:', i, 'Loss:', loss.item())

        torch.save(self.net1.state_dict(), './model_w.pkl')

    # ————————————用模型预测未来数据————————————
    def prediction(self, offset, predict_number):
        end2 = len(self.train_x) - self.distance_test - offset  # 扣除offset的起始点位置
        prev_seq = self.train_x[end2]  # 预测起始点的输入
        label = []  # 测试标签
        pre_predict = []  # 记录用训练数据预测的结果，数据无意义，仅用于隐含层记忆历史数据
        predict = []  # 有效的预测结果

        prev_seq = torch.Tensor(prev_seq)
        for i in range(offset + predict_number):
            prev_seq = torch.unsqueeze(prev_seq, 0)

            next_seq = self.net1(prev_seq)
            if i < offset:
                pre_predict.append(next_seq[-1][-1][-1].detach().numpy())
                prev_seq = self.train_y[end2 + i]
            else:
                predict.append(next_seq[-1][-1][-1].detach().numpy())
                prev_seq = torch.squeeze(prev_seq, dim=0)
                prev_seq = torch.cat((prev_seq[1:], next_seq[:, -1]))

        predict = np.array(predict)
        pre_predict = np.array(pre_predict)

        # 数据复原
        predict = predict * self.std_data + self.mean_data
        pre_predict = pre_predict * self.std_data + self.mean_data

        return predict

    # def draw(self, label, pre_predict, predict):
    #     fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    #     if label:
    #         ax.plot(list(range(len(label))), label, color='b')
    #     if pre_predict:
    #         ax.plot(list(range(len(pre_predict))), pre_predict, color='r')
    #     if predict:
    #         ax.plot(list(range(len(pre_predict), len(pre_predict) + len(predict))), predict, color='y')
    #
    #     if label and pre_predict and predict:
    #         xlim_min = 1
    #         xlim_max = len(pre_predict) + len(predict)
    #         ylim_min = min(min(label), min(pre_predict), min(predict))
    #         ylim_max = max(max(label), max(pre_predict), max(predict))
    #     else:
    #         # Default limits if any of the arrays is empty
    #         xlim_min, xlim_max, ylim_min, ylim_max = 0, 1, 0, 1
    #
    #     ax.set_xlim(xlim_min, xlim_max)
    #     ax.set_ylim(ylim_min, ylim_max)
    #     plt.show()

def QandA():
    distance_test = 0  # 训练数据的截止点离最新数据的距离
    predict_number = 5  # 连续预测天数
    offset = 30

    instance_train_prediction = train_prediction(distance_test)
    instance_train_prediction.train_lstm()
    prediction_result = instance_train_prediction.prediction(offset, predict_number)
    print("Prediction result:", prediction_result)

if __name__ == '__main__':
    QandA()

