import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis, KernelPCA
from sklearn.preprocessing import MinMaxScaler
from factor_analyzer import FactorAnalyzer

class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, encoding='gbk')
        self.data = self.data.fillna(0)
        self._process_data()

    def _process_data(self):
        self.data['close'] = self.data['close'].str.replace(',', '').astype(float)
        self.data['open'] = self.data['open'].str.replace(',', '').astype(float)
        self.data['high'] = self.data['high'].str.replace(',', '').astype(float)
        self.data['low'] = self.data['low'].str.replace(',', '').astype(float)
        self.data['amt'] = self.data['amt'].astype(float)
        self.data['MA10'] = self.data['MA10'].astype(float)
        self.data['MA20'] = self.data['MA20'].astype(float)
        self.data['DIF'] = self.data['DIF'].astype(float)
        self.data['DEA'] = self.data['DEA'].astype(float)
        self.data['MACD'] = self.data['MACD'].astype(float)
        self.data['K'] = self.data['K'].astype(float)
        self.data['D'] = self.data['D'].astype(float)
        self.data['J'] = self.data['J'].astype(float)
        self.data['RSI'] = self.data['RSI'].astype(float)
        self.data['WR'] = self.data['WR'].astype(float)
        self.data['WR1'] = self.data['WR1'].astype(float)
        self.data['BIAS1'] = self.data['BIAS1'].astype(float)
        self.data['BIAS2'] = self.data['BIAS2'].astype(float)
        self.data['BIAS3'] = self.data['BIAS3'].astype(float)
        self.data['PSY'] = self.data['PSY'].astype(float)
        self.data['PSYMA'] = self.data['PSYMA'].astype(float)
        self.data['CCI'] = self.data['CCI'].astype(float)
        self.data['ATR'] = self.data['ATR'].astype(float)
        self.data['BBI'] = self.data['BBI'].astype(float)

        self.data1 = self.data[['close', 'open', 'high', 'low','amt', 'MA5','MA10',
                 'MA20','DIF','DEA','MACD','K','D','J','RSI','WR','WR1',
                 'BIAS1','BIAS2','BIAS3','PSY','PSYMA','CCI','ATR','BBI',
                 ]].values
        self.features = self.data[['close', 'open', 'high', 'low','amt', 'MA5','MA10',
                 'MA20','DIF','DEA','MACD','K','D','J','RSI','WR','WR1',
                 'BIAS1','BIAS2','BIAS3','PSY','PSYMA','CCI','ATR','BBI',
                 ]].values

        # 应用 MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = scaler.fit_transform(self.features)

        # PCA
        pca = PCA(n_components=0.95)
        pca.fit(self.features)
        self.n_pca_components = pca.n_components_
        self.pca_cumulative_variance_ratio = np.sum(pca.explained_variance_ratio_) * 100
        self.features_pca = pca.transform(self.features)
        pca_components_restored = pca.inverse_transform(self.features_pca)
        pca_restored = scaler.inverse_transform(pca_components_restored)
        for i in range(self.n_pca_components):
            self.data[f'pca_{i + 1}_restored'] = pca_restored[:, i]


        # Factor Analysis
        n_components = 4
        fa = FactorAnalyzer(n_factors=n_components, rotation="varimax")
        fa.fit_transform(self.features)
        self.n_fa_components = n_components
        self.features_fa = fa.transform(self.features)

        # 反归一化FA
        fa_components_restored = np.dot(self.features_fa, fa.loadings_.T)
        fa_restored = scaler.inverse_transform(fa_components_restored)

        # 将反归一化后的值限制在原始数据的范围内
        fa_restored = np.maximum(np.minimum(fa_restored, np.max(self.data1, axis=0)), np.min(self.data1, axis=0))

        for i in range(self.n_fa_components):
            self.data[f'fa_{i + 1}_restored'] = fa_restored[:, i]

    def get_data_pca(self):
        return self.data.filter(like='pca')

    def get_data_fa(self):
        # # 选择以 "kpca" 开头的列
        # fa_data = self.data.filter(like='fa')
        # # 插入 "close" 列到第一列
        # fa_data.insert(0, 'close', self.data['close'])
        # return fa_data
        return self.data.filter(like='fa')

# # Usage
# processor = DataProcessor("shangzheng_factor.csv")
# data_pca = processor.get_data_pca()
# data_pca = np.array(data_pca)
#
# data_fa = processor.get_data_fa()
# data_fa = np.array(data_fa)
#
# print(data_pca)
# print(data_fa)