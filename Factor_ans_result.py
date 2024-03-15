import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import seaborn as sns


# 读取数据
data = pd.read_csv("maotai_factor1.csv", encoding='gbk')

# 填充缺失值
data = data.fillna(0)

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

# 提取特征
features = data[['close', 'open', 'high', 'low','amt', 'MA5','MA10',
                 'MA20','DIF','DEA','MACD','K','D','J','RSI','WR','WR1',
                 'BIAS1','BIAS2','BIAS3','PSY','PSYMA','CCI','ATR','BBI',
                 ]].values


# 应用 MinMaxScaler
# # 找到特征矩阵中的最大值和最小值
# min_values = np.min(features, axis=0)
# max_values = np.max(features, axis=0)
#
# # 使用逻辑来进行归一化
# features = (features - min_values) / (max_values - min_values)
#
# # 保留最大最小值
# min_values_saved = min_values
# max_values_saved = max_values

scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features)

# print(features.shape)
# PCA

pca = PCA(n_components=0.95)
pca.fit(features)
n_pca_components = pca.n_components_
pca_cumulative_variance_ratio = np.sum(pca.explained_variance_ratio_) * 100
X_new = pca.transform(features)
features_pca = pca.transform(features)
# 替换原特征
for i in range(n_pca_components):
    data[f'pca_{i+1}'] = features_pca[:, i]


# KPCA
# kpca = KernelPCA(kernel='rbf',gamma=0.01)  # No need to specify n_components here
# features_kpca = kpca.fit_transform(features)
# explained_variance_ratio_kpca = kpca.lambdas_ / np.sum(kpca.lambdas_)
# cumulative_variance_ratio_kpca = np.cumsum(explained_variance_ratio_kpca)
# n_kpca_components = np.argmax(cumulative_variance_ratio_kpca >= 0.95) + 1
# kpca_cumulative_variance_ratio = cumulative_variance_ratio_kpca[n_kpca_components - 1] * 100
# features_kpca = features_kpca[:, :n_kpca_components]

# Replace original features with Kernel PCA components
# for i in range(n_kpca_components):
#     data[f'kpca_{i+1}'] = features_kpca[:, i]


# Factor Analysis
n_components = 4
fa = FactorAnalyzer(n_factors=n_components, rotation="varimax")
fa.fit_transform(features)
n_fa_components = n_components
features_fa = fa.transform(features)

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

# Replace original features with Factor Analysis factors
for i in range(n_fa_components):
    data[f'fa_{i+1}'] = features_fa[:, i]


data_pca= data.filter(like='pca')
data_pca=np.array(data_pca)
print("\nPCA分析数据集：")
print(data.filter(like='pca').head())
print(data_pca)

# data_kpca= data.filter(like='kpca')
# data_kpca=np.array(data_kpca)
# print("\nKPCA数据集：")
# # print(data.filter(like='kpca').head())
# print(data_kpca)

data_fa= data.filter(like='fa')
data_fa=np.array(data_fa)
print("\n因子分析数据集：")
print(data.filter(like='fa').head())
print(data_fa)


# 输出结果
print(f"通过主成分分析法提取了{n_pca_components}个主成分分量，累计信息贡献率为{pca_cumulative_variance_ratio:.2f}%。")
# print(f"通过核主成分分析法（KPCA）提取了{n_kpca_components}个主成分分量，累计信息贡献率为{kpca_cumulative_variance_ratio:.2f}%。")
# print(f"通过因子分析法提取了{n_fa_components}个因子，累计信息贡献率为{fa_cumulative_variance_ratio:.2f}%。")


