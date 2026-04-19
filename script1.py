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
    '早峰前时段',               # 条件1为真时的值
    np.where(
        df_type0['hour'] >= 22,   # 条件2：深夜时段
        '深夜时段',                # 条件2为真时的值
        '其他时段'                 # 两个条件都不满足时的值
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


## 24小时刷卡量分布可视化
#按小时统计刷卡量
hourly_counts = df.groupby('hour').size()
hourly_counts = hourly_counts.reindex(range(24), fill_value=0)

fig, ax = plt.subplots(figsize=(14, 6))
colors = []
for hour in range(24):
    if hour < 7:
        colors.append('#98C3C7')  #浅蓝色：早峰前时段
    elif hour >= 22:
        colors.append('#5399A0')  #蓝色：深夜时段
    else:
        colors.append('#DDB355')  #黄色：其他时段

bars = ax.bar(range(24), hourly_counts.values, color=colors, linewidth=0.5)

# 设置x轴
ax.set_xticks(range(24))
ax.set_xticklabels(range(24), fontsize=10)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel('小时', fontsize=12, fontweight='bold')

# 设置y轴
ax.set_ylabel('刷卡量（次）', fontsize=12, fontweight='bold')
ax.yaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
ax.set_axisbelow(True)

# 标题
ax.set_title('24小时刷卡量分布', fontsize=16, fontweight='bold', pad=20)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#98C3C7', label='早峰前时段 (hour < 7)'),
    Patch(facecolor='#5399A0', label='深夜时段 (hour ≥ 22)'),
    Patch(facecolor='#DDB355', label='其他时段 (7 ≤ hour < 22)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

plt.savefig('hour_distribution.png', dpi=150, bbox_inches='tight')
plt.show()