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


# task3 线路站点分析
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame
    包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列"""
    # 按线路号分组，计算均值和标准差
    route_stats = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    # 重命名列
    route_stats.columns = ['线路号', 'mean_stops', 'std_stops']
    # 处理可能的 NaN 值（当某线路只有一条记录时，std 为 NaN）
    route_stats['std_stops'] = route_stats['std_stops'].fillna(0)
    # 按 mean_stops 降序排列
    route_stats = route_stats.sort_values('mean_stops', ascending=False).reset_index(drop=True)
    return route_stats

route_stats_df = analyze_route_stops(df)
print("各线路平均搭乘站点数及标准差（前10行）：")
print(route_stats_df.head(10).to_string(index=False))
print()

## 使用 seaborn barplot 绘制水平条形图

# 取均值最高的前15条线路
top15_routes = route_stats_df.head(15).copy()
# 转换为字符串类型
top15_routes['线路号'] = top15_routes['线路号'].astype(str)
# 按mean_stops升序排列
top15_routes = top15_routes.sort_values('mean_stops', ascending=True)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8))

bars = sns.barplot(
    data=top15_routes,
    x='mean_stops',        # 平均搭乘站点数
    y='线路号',             # 线路号
    hue='线路号',           # 将y变量赋值给 hue
    legend=False,
    ax=ax,
    palette='Blues_d',
    err_kws={
        'color': 'black',
        'linewidth': 1.5
    },
    capsize=0.3
)

# 添加误差棒（显示标准差）
for i, (_, row) in enumerate(top15_routes.iterrows()):
    mean_val = row['mean_stops']
    std_val = row['std_stops']
    # 绘制水平误差棒
    ax.errorbar(
        x=mean_val,
        y=i,
        xerr=std_val,
        capsize=0.3,
        color='black',
        linewidth=1.0
    )

# 计算最大均值+最大标准差，作为x轴的上限
max_mean = top15_routes['mean_stops'].max()
max_std = top15_routes.loc[top15_routes['mean_stops'].idxmax(), 'std_stops']
x_max = max_mean + max_std + 3  # 留出边距
ax.set_xlim(0, x_max)

# 设置标题和标签
ax.set_title('平均搭乘站点数最高的前15条线路', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('平均搭乘站点数', fontsize=12, fontweight='bold')
ax.set_ylabel('线路号', fontsize=12, fontweight='bold')

# 添加网格线
ax.xaxis.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
ax.set_axisbelow(True)

# 添加数值标签
for i, (_, row) in enumerate(top15_routes.iterrows()):
    mean_val = row['mean_stops']
    std_val = row['std_stops']
    ax.text(
        mean_val + std_val + 0.1,
        i,
        f'  {mean_val:.2f} ± {std_val:.2f}',
        va='center',
        fontsize=9
    )

# 去掉上右的边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('route_stops.png', dpi=150, bbox_inches='tight')
plt.show()