import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


# ——————————————————————————————数据预处理——————————————————————————————————————————
# 读取数据
data = pd.read_csv("maotai_factor1.csv", encoding='gbk')
data = data.fillna(0)

# 数据表转换为浮点数
data['close'] = data['close'].str.replace(',', '').astype(float)
data['open'] = data['open'].str.replace(',', '').astype(float)
data['high'] = data['high'].str.replace(',', '').astype(float)
data['low'] = data['low'].str.replace(',', '').astype(float)

data[['amt', 'MA5', 'MA10', 'MA20', 'DIF', 'DEA', 'MACD', 'K', 'D', 'J', 'RSI',
      'WR', 'WR1', 'BIAS1', 'BIAS2', 'BIAS3', 'PSY', 'PSYMA', 'CCI', 'ATR',
      'BBI']] = data[['amt', 'MA5', 'MA10', 'MA20', 'DIF', 'DEA', 'MACD', 'K',
                      'D', 'J', 'RSI', 'WR', 'WR1', 'BIAS1', 'BIAS2', 'BIAS3',
                      'PSY', 'PSYMA', 'CCI', 'ATR', 'BBI']].astype(float)

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['date']))
# -------------------------------------end-------------------------------------------

# ————————————————————————————————————————————聚类热图————————————————————————————————————————
# 使用K均值聚类进行特征聚类
# kmeans = KMeans(n_clusters=11, random_state=42)
# cluster_labels = kmeans.fit_predict(scaled_features)
#
# # 添加聚类标签到数据
# data['cluster'] = cluster_labels
#
# # 提取聚类后的特征
# clustered_features = data[['close','open','high','low',
#                            'BBI', 'MA5', 'MA10', 'MA20', 'DIF', 'DEA', 'MACD', 'K', 'D', 'J', 'RSI',
#                            'WR', 'WR1', 'BIAS1', 'BIAS2', 'BIAS3', 'PSY', 'PSYMA', 'CCI', 'ATR', 'amt','cluster']]
#
# # 绘制简化的聚类热图
# plt.figure(figsize=(12, 10))
# sns.heatmap(clustered_features.groupby('cluster').mean(), cmap='viridis', annot=False)
# plt.title('Clustered Features Heatmap')
# plt.show()
# -----------------------end--------------------------------------


# ——————————————————————————————相关性热力图————————————————————————————————————————
# 数据预处理
# numeric_cols = ['close', 'open', 'high', 'low', 'amt', 'MA5', 'MA10', 'MA20', 'DIF', 'DEA', 'MACD', 'K', 'D', 'J', 'RSI', 'WR', 'WR1', 'BIAS1', 'BIAS2', 'BIAS3', 'PSY', 'PSYMA', 'CCI', 'ATR', 'BBI']
# data[numeric_cols] = data[numeric_cols].replace(',', '', regex=True).astype(float)

# 计算相关性矩阵
correlation_matrix = data.corr()

# 绘制热力图
plt.figure(figsize=(16, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('')
plt.tight_layout()
plt.savefig('heatmap1.png', dpi=400, bbox_inches='tight')
# -----------------------end--------------------------------------