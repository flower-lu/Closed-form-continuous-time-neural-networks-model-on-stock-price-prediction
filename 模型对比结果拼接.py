import pandas as pd

# 读取CSV文件
df = pd.read_csv('predicted_values2.csv', header=None)

# 去除[]并转换为浮点数
df = df.applymap(lambda x: float(x.strip('[]')))

# 将处理后的数据复制到新列
for col in df.columns:
    df[f'predictions_{col}'] = df[col]

# 写入新的CSV文件
df.to_csv('predictions1.csv', index=False)
