import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


### task1 数据预处理

# 读取数据
df = pd.read_csv('ICData.csv', sep=',')
print(df.head(),'\n')  #打印前5行
print("详细数据集信息:")
print(df.info())
print()

# 时间解析
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour

# 构造衍生字段
df['ride_stops'] = abs(df['下车站点'] - df['上车站点'])
rows_before = len(df)   #记录删除前的行数
df = df[df['ride_stops'] != 0]   #仅保留ride_stops不为0的行
print('删除行数为：%d\n'%(rows_before - len(df)))    #打印删除行数

#缺失值检查
print("各列缺失值数量：")
missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    df = df.dropna()
    rows_deleted = rows_before - len(df)
    print(f"\n存在缺失值，已删除 {rows_deleted} 行")
else:
    print("0\n无缺失值，无需删除行\n")
