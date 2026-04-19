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


### task2 时间分布分析

##早晚时段刷卡量统计

df_type0 = df[df['刷卡类型'] == 0]
total_type0 = len(df_type0)
# 使用 np.where 创建时段标识列（在筛选后的数据上操作）
df_type0 = df_type0.copy()
df_type0['time_period'] = np.where(
    df_type0['hour'] < 7,     # 条件1：早峰前时段
    '早峰前时段',              # 条件1为真时的值
    np.where(
        df_type0['hour'] >= 22,  # 条件2：深夜时段
        '深夜时段',               # 条件2为真时的值
        '其他时段'                # 两个条件都不满足时的值
    )
)
# 统计各时段刷卡量
morning_type0 = np.where(
    df_type0['time_period'] == '早峰前时段',
    1,
    0
).sum()
night_type0 = np.where(
    df_type0['time_period'] == '深夜时段',
    1,
    0
).sum()
#计算占比
morning_pct = (morning_type0 / total_type0) * 100 if total_type0 > 0 else 0#计算占比
night_pct = (night_type0 / total_type0) * 100 if total_type0 > 0 else 0
print(f"早峰前时段（交易时间早于 07:00）刷卡量为{morning_type0}，占全天的{morning_pct:.2f}%")
print(f"深夜时段（交易时间晚于 22:00）刷卡量为{night_type0}，占全天的{night_pct:.2f}%")
print()