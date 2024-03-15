import pandas as pd

# 读取CSV文件
df = pd.read_csv('predicted_vs_true_prices.csv')

# 初始化label列
df['label'] = 0

# 遍历数据，确定label的值
for i in range(1, len(df)):
    if df.loc[i, 'Predicted Price'] > df.loc[i-1, 'Predicted Price']:
        df.loc[i, 'label'] = 1

# 将结果保存到新的CSV文件
df.to_csv('output_file.csv', index=False)
