# 股票价格预测的闭式连续时间模型研究(Research on Closed-form Continuous-time Models for stock price prediction)

闭式连续时间神经网络（CfC）是一种功能强大的连续液体神经信息处理单元。

本模型基于Tensorflow架构，需要配置对应版本的keras和cuda库

## Requirements

- Python--3.6 or newer
- Tensorflow--2.4 or newer
- scikit-learn--0.24.2 or newer
- matplotlib--3.6.0               
- numpy--1.19.5
- pandas--1.2.1
- cudatoolkit--11.0.221
- cudnn--8.9.2.26
- keras--2.4.3

## Module description

- ```tf_cfc.py``` 主CfC模型 (在 Tensorflow 2.x 中实现 CfC）
- ```tf_cfc_gru.py``` 使用了gru模块的CfC模型
- ```tf_cfc_resnet.py``` 使用了残差块的CfC模型
- ```cfc_train.py``` 主训练模型
- ```cfc_t+n.py``` 对数据集的T+N（N=5，10，15，20）日后的涨跌情况进行了预测
- ```other_train.py``` 使用CNN,RNN,LSTM对数据集进行训练
- ```revenue_stock.py``` 盈利能力评估
- ```Factor_ans_result.py```特征辨识的结果
- ```Factor_ans_use1.py``` 特征辨识的调用文件
- ```FA.py``` 因子分析法部分的文件
- ```PCA.py``` 主成分分析法部分的文件
- ```cfc_ts_solo_pic``` 有tushare进行股票数据更改的cfc股价预测模型
- ```cfc_01.py``` 生成混淆矩阵和测试集的预测真实值文件
- ```updown_label.py``` 根据cfc_01.py 生成的文件生成0,1涨跌标签
- ```heatmap_cluster.py``` 聚类热图和相关性热图
- ```maotai_factor1.csv``` 茅台股票数据
- ```shangzheng_factor1.csv``` 上证综合指数数据

## Usage

结合论文《基于闭式连续时间神经网络的股票预测模型》

## Cite

```
@article{hasani_closed-form_2022,
	title = {Closed-form continuous-time neural networks},
	journal = {Nature Machine Intelligence},
	author = {Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Liebenwein, Lucas and Ray, Aaron and Tschaikowski, Max and Teschl, Gerald and Rus, Daniela},
	issn = {2522-5839},
	month = nov,
	year = {2022},
}
```
