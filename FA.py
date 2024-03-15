# Import required libraries
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from sklearn import preprocessing
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import seaborn as sns
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")  # 忽略一些警告 不影响运行

data= pd.read_csv("maotai_factor.csv")
data = data.fillna(0)
data = data.drop('date',axis=1)
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

x=data


min_max_scaler = preprocessing.MinMaxScaler()
x_minMax = min_max_scaler.fit_transform(x)#归一化
#
# #mean of each feature
n_samples, n_features = x_minMax.shape
# mean=np.array([np.mean(x_minMax[:,i]) for i in range(n_features)])  #normalization
# x = x_minMax-mean  #去中心化
x = x_minMax
x = pd.DataFrame(x)

# chi_square_value,p_value=calculate_bartlett_sphericity(x)
# print(chi_square_value)
# print(p_value)
#
# kmo_all,kmo_model=calculate_kmo(x)
# print(kmo_model)

# Dropping unnecessary columns
x.drop([],axis=1,inplace=True)
# Dropping missing values rows
x.dropna(inplace=True)
x.head()

# x = np.nan_to_num(x)
#  充分性测试(Adequacy Test) kmo>0.6,相关矩阵不是一个identity matrix


# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(5, rotation="varimax")
# print(x)
fa.fit(x)

# Check Eigenvalues
ev, v = fa.get_eigenvalues()

# Create scree plot using matplotlib
# plt.scatter(range(1,x.shape[1]+1),ev)
# plt.plot(range(1,x.shape[1]+1),ev)
# plt.title('碎石图')
# plt.xlabel('因子')
# plt.ylabel('特征值')
# # Set the major ticks on x and y axes to 1 unit apart
# plt.xticks(np.arange(1, x.shape[1]+1, 1))
# plt.yticks(np.arange(min(ev), max(ev)+1, 1))
# plt.grid()
# plt.show()
#
# print(fa.loadings_)

# df_cm = pd.DataFrame(np.abs(fa.loadings_), index=x.columns)
# plt.figure(figsize = (20,20))
# ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
# # 设置y轴的字体的大小
# ax.yaxis.set_tick_params(labelsize=15)
# plt.title('因子载荷热图', fontsize='xx-large')
# # Set y-axis label
# plt.ylabel('技术指标', fontsize='xx-large')
# plt.savefig('factorAnalysis11.png', dpi=500)
#
# fa.transform(x)
# plt.show()

df_cm = pd.DataFrame(np.abs(fa.loadings_), index=x.columns)
# 设置字体大小
# 创建热图
plt.figure(figsize=(6, 12))
ax = sns.heatmap(df_cm, annot=True, cmap="BuPu")
ax.set_yticklabels(ax.get_yticklabels(), size=6)
# 设置横坐标标签为数字
ax.set_xticklabels(np.arange(1, len(df_cm.columns) + 1))
# 设置标题和标签
plt.title('', fontsize=10)
plt.ylabel('技术指标', fontsize=10)
plt.xlabel('因子', fontsize=10)
# 保存图像并显示
plt.savefig('factorAnalysis11.png', dpi=500)
# plt.show()


# Get factor variance
factor_variance = fa.get_factor_variance()
# 获取主成分数量
num_factors = len(factor_variance[0])

# 获取累计信息贡献率
cumulative_variance_ratio = np.cumsum(factor_variance[1])

# 打印主成分数量和累计信息贡献率
print('因子数量:', num_factors)
print('累计信息贡献率:')
for i in range(num_factors):
    print(f'Factor {i+1}: {cumulative_variance_ratio[i]*100:.2f}%')
