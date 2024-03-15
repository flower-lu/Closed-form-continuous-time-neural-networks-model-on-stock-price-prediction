import pandas as pd

# 读取CSV文件
data = pd.read_csv('predict_buy_self.csv')

# 将日期列转换为日期时间格式
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')

# 设定起始资金
initial_capital = 100000  # ¥

# 交易手续费
transaction_fee_rate = 0.001  # 0.1%

# 初始化交易次数和交易记录列表
num_trades = 0
trade_records = []
current_holdings = 0  # 当前持有股票数
current_assets = initial_capital  # 当前资产
last_buy_price = None  # 上次购买价格

# 将标签列向上移动一行
data['label'] = data['label'].shift(-1)
# 模拟交易
for index, row in data.iterrows():
    if index == 0:  # Skip first row
        continue

    # 如果预测为1，表示股价会上涨，则买入
    if row['label'] == 1 and current_assets > 0 and current_holdings == 0:
        buy_date = row['date']
        buy_price = row['close']
        last_buy_price = buy_price
        max_shares = current_assets // buy_price
        num_shares = min(max_shares, current_assets // buy_price)
        total_cost = num_shares * buy_price
        transaction_fee = total_cost * transaction_fee_rate

        num_trades += 1
        current_holdings += num_shares
        current_assets -= total_cost + transaction_fee

        trade_records.append([buy_date, buy_price, num_shares, None, None, transaction_fee, None , None])
        # trade_records[-1][5] = transaction_fee
    # 如果预测为0，表示股价会下跌，则卖出
    elif row['label'] == 0 and current_holdings > 0:
        sell_date = row['date']
        sell_price = row['close']
        total_value = current_holdings * sell_price
        transaction_fee = total_value * transaction_fee_rate

        profit = total_value - (current_holdings * last_buy_price)  # Calculate profit based on last buy price

        num_trades += 1
        current_assets += total_value - transaction_fee + profit
        current_holdings = 0

        trade_records[-1][3] = sell_date
        trade_records[-1][4] = sell_price
        trade_records[-1][6] = transaction_fee
        trade_records[-1][7] = profit

# 将交易记录转换为DataFrame
trade_df = pd.DataFrame(trade_records, columns=['买入日期', '买入单价/¥', '买入数量', '卖出日期', '卖出单价/¥', '买入手续费/¥', '卖出手续费/¥', '收益/¥'])

# 输出交易次数
print(f"一共产生了{num_trades}次交易")

# 输出交易记录表
print(trade_df)

# 输出最终资金
final_assets = initial_capital + trade_df['收益/¥'].sum() - trade_df['买入手续费/¥'].sum() - trade_df['卖出手续费/¥'].sum()
print(f"最终资金: {final_assets}¥")

# 将交易记录导出为CSV文件
trade_df.to_csv('trade_records.csv', index=False)