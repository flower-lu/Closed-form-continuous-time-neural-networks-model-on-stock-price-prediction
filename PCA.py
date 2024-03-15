import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.preprocessing import MinMaxScaler
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")  # 忽略一些警告 不影响运行
# 读取数据
data = pd.read_csv("shangzheng_factor.csv",encoding='gbk')
data = data.fillna(0)
# print(data)
# print(data.columns)

data_date = data[['date']]
data_date = np.array(data_date)

date_scaler = MinMaxScaler(feature_range=(0, 1))
x_date_train_normalized = date_scaler.fit_transform(np.array(data_date).reshape(-1, 1))

# Reshape back to original shape
x_date_train_resc = x_date_train_normalized.reshape(data_date.shape)
x_date_train = x_date_train_resc

# print(x_date_train[:,0].shape)
# exit()
# 数据预处理
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

# 提取特征
features = data[['close', 'open', 'high', 'low','amt', 'MA5','MA10',
                 'MA20','DIF','DEA','MACD','K','D','J','RSI','WR','WR1',
                 'BIAS1','BIAS2','BIAS3','PSY','PSYMA','CCI','ATR','BBI',
                 ]].values
#输入数据
x_close = data[['close']]
x_close = np.array(x_close)

x_scaler = MinMaxScaler(feature_range=(0, 1))
x_close_normalized = x_scaler.fit_transform(np.array(x_close).reshape(-1, 1))

# Reshape back to original shape
x_close_resc = x_close_normalized.reshape(x_close.shape)
x_close = x_close_resc
#
# Save the
data.to_excel("data.xlsx", index=False)
#
# Load data from Excel file
data = openpyxl.load_workbook('data.xlsx')
table = data.active
#
# Extract data from Excel sheet into a numpy array
x = []
for row in table.iter_rows(min_row=3, max_row=2437, min_col=2, max_col=26, values_only=True):
    x.append(row)
#
# Convert the list of lists to a numpy array
x = np.array(x, dtype=float)
#
# Handle missing values (replace NaN with zeros)
x = np.nan_to_num(x)
#
#
# Handle infinite values (replace +/- infinity with large finite numbers)
x[np.isinf(x)] = np.finfo(np.float64).max
#
# Min-max scaling
min_max_scaler = preprocessing.MinMaxScaler()
x_minMax = min_max_scaler.fit_transform(x)
#
# # Calculate mean of each feature
mean = np.mean(x_minMax, axis=0)
#
# Centering the data
# xd = x_minMax - mean
xd = x_minMax

# print(x[:,0].shape)
# exit()

scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features)
xd = features
# Perform PCA
model = PCA(n_components=0.95)
model.fit(xd)
X_new = model.transform(xd)
Maxcomponent = model.components_
ratio = model.explained_variance_ratio_
score = model.score(xd)

# Display results
print('降维后的数据:', X_new)
print('返回具有最大方差的成分:', Maxcomponent)
print('保留主成分的方差贡献率:', ratio)
print('所有样本的log似然平均值:', score)
print('奇异值:', model.singular_values_)
print('噪声协方差:', model.noise_variance_)

# Calculate the confidence ellipses
# def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
#     cov = np.cov(x, y)
#     pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
#     eigenvalues, eigenvectors = np.linalg.eigh(cov)
#     ellipse_radius_x = np.sqrt(eigenvalues[0]) * n_std
#     ellipse_radius_y = np.sqrt(eigenvalues[1]) * n_std
#     ellipse = Ellipse((np.mean(x), np.mean(y)), width=ellipse_radius_x * 2, height=ellipse_radius_y * 2,
#                       facecolor=facecolor, **kwargs)
#     ax.add_patch(ellipse)
#     return ax
#
#
# # Plot PCA scores with different colors for each principal component
# fig, ax = plt.subplots()
# # Adding transparency grid
# ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
# # 遍历每个主成分并绘制散点图
# colors = ['red', 'green', 'blue', 'yellow', 'purple','pink','orange']
# for i in range(5):
#     ax.scatter(X_new[:, 0], X_new[:, 1], c=X_new[:, i], cmap=plt.cm.get_cmap('viridis'), alpha=0.6, label=f'PC{i+1}')
#
# # 添加置信度椭圆
# for i in range(5):
#     confidence_ellipse(X_new[i*200:(i+1)*200, 0], X_new[i*200:(i+1)*200, 1], ax, edgecolor='red',facecolor=plt.cm.get_cmap('viridis')(i/5),alpha=0.2)
#
# # 绘制图例
# plt.legend()
#
# # Plotting axes
#
# plt.xlabel('Dim1 (Score: {:.2f}%)'.format(model.explained_variance_ratio_[0] * 100))
# plt.ylabel('Dim2 (Score: {:.2f}%)'.format(model.explained_variance_ratio_[1] * 100))
# plt.title('带有置信椭圆的 PCA 得分图')
# plt.axis('equal')  # Make the plot axes equally scaled
#
# # Removing frame
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)



# scale_factor = 5.5
# for i in range(5):
#     ax.arrow(0, 0, model.components_[0, i] * scale_factor, model.components_[1, i] * scale_factor, color='black', alpha=0.8,
#              linewidth=2, head_width=0.05, head_length=0.05)
#     ax.text(model.components_[0, i] * scale_factor * 1.15, model.components_[1, i] * scale_factor * 1.15, 'PC{}'.format(i + 1),
#             color='g', ha='center', va='center')


# plt.show()

# 获取主成分数量
num_components = model.n_components_

# 获取累计信息贡献率
cumulative_var_ratio = np.cumsum(model.explained_variance_ratio_)

# 打印主成分数量和累计信息贡献率
print('主成分数量:', num_components)
print('累计信息贡献率:')
for i in range(num_components):
    print(f'PC{i+1}: {cumulative_var_ratio[i]*100:.2f}%')





# Visualize the reduced data
plt.figure(figsize=(10, 6))
plt.scatter(x_close, X_new[:, 0], c='purple', label='PC1', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
plt.scatter(x_close, X_new[:, 1], c='r', label='PC2', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
plt.scatter(x_close, X_new[:, 2], c='b', label='PC3', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
plt.scatter(x_close, X_new[:, 3], c='g', label='PC4', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
plt.scatter(x_close, X_new[:, 4], c='y', label='PC5', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
plt.scatter(x_close, X_new[:, 5], c='pink', label='PC6', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
plt.scatter(x_close, X_new[:, 6], c='orange', label='PC7', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
plt.xlabel('收盘价特征')
plt.ylabel('主成分')
plt.title('')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # 添加透明度网格
plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))  # 将图例移到右侧
plt.savefig('factorAnalysis11.png', dpi=500)
# plt.show()
#
#
# plt.figure(figsize=(8, 6))
# plt.scatter(x[:, 0], x[:, 0], c='purple', label='PC1', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
# plt.scatter(x[:, 0], x[:, 1], c='r', label='PC2', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
# plt.scatter(x[:, 0], x[:, 2], c='b', label='PC3', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
# plt.scatter(x[:, 0], x[:, 3], c='g', label='PC4', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
# plt.scatter(x[:, 0], x[:, 4], c='y', label='PC5', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.6)
# plt.scatter(x[:, 0], x[:, 5], c='cyan', label='PC6', cmap=plt.cm.Set1, edgecolor='none', s=40, alpha=0.4)
# plt.xlabel('收盘价特征')
# plt.ylabel('主成分')
# plt.title('降维前散点图')
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # 添加透明度网格
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))  # 将图例移到右侧
# plt.show()
#
#
#
#
#
#
# Visualize the reduced data in 3D。
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(x_close, X_new[:, 0],x_date_train, c='purple', label='PC1', cmap=plt.cm.Set1, edgecolor='none', marker='o', depthshade=True, s=40, alpha=0.4)
# ax.scatter(x_close, X_new[:, 1], x_date_train, c='r', label='PC2', cmap=plt.cm.Set1, edgecolor='none', marker='o', depthshade=True, s=40, alpha=0.4)
# ax.scatter(x_close, X_new[:, 2], x_date_train, c='b', label='PC3', cmap=plt.cm.Set1, edgecolor='none', marker='o', depthshade=True, s=40, alpha=0.4)
# ax.scatter(x_close, X_new[:, 3], x_date_train, c='g', label='PC4', cmap=plt.cm.Set1, edgecolor='none', marker='o', depthshade=True, s=40, alpha=0.4)
# ax.scatter(x_close, X_new[:, 4], x_date_train, c='y', label='PC5', cmap=plt.cm.Set1, edgecolor='none', marker='o', depthshade=True, s=40, alpha=0.4)
# ax.scatter(x_close, X_new[:, 5],x_date_train,  c='pink', label='PC6', cmap=plt.cm.Set1, edgecolor='none', marker='o', depthshade=True, s=40, alpha=0.4)
# ax.scatter(x_close, X_new[:, 6], x_date_train, c='orange', label='PC7', cmap=plt.cm.Set1, edgecolor='none', marker='o', depthshade=True, s=40, alpha=0.4)
#
#
# ax.set_xlabel('收盘价特征')
# ax.set_ylabel('时间')
# ax.set_zlabel('主成分')
# ax.set_title('')
# legend = ax.legend()
# for handle in legend.legendHandles:
#     handle.set_sizes([30])
#     handle.set_alpha(0.6)
# plt.savefig('factorAnalysis11.png', dpi=500)
#
# plt.show()





