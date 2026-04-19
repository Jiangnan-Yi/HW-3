# 易江南-25361043-第三次人工智能编程作业

## 1. 任务拆解与 AI 协作策略
将6项分析任务分步拆解给 AI，先让 AI 做数据读取，再做可视化

## 2. 核心 Prompt 迭代记录
初代 Prompt：
使用 seaborn 水平条形图（barplot） 对analyze_route_stops函数返回的结果（取均值最高的前15条线路）进行可视化，要求：
○	误差棒（errorbar）显示标准差，capsize=0.3
○	颜色使用 seaborn palette（如 "Blues_d"）
○	包含中文标题、x/y 轴中文标签，x 轴范围从 0 起始
○	图像保存为 route_stops.png（dpi=150）
AI 生成的问题：误差棒超出了图表的范围导致显示不完全
优化后的 Prompt：
使用 seaborn 水平条形图（barplot） 对analyze_route_stops函数返回的结果（取均值最高的前15条线路）进行可视化，要求：
○	误差棒（errorbar）显示标准差，capsize=0.3
○	颜色使用 seaborn palette（如 "Blues_d"）
○	包含中文标题、x/y 轴中文标签，x 轴范围从 0 起始
○	图像保存为 route_stops.png（dpi=150）
○	调整x轴长度，使误差棒能够显示完全

## 3. Debug 记录
报错现象：Keyerror，没有提前创建'minute'列
解决过程：让AI生成以下代码：判断'minute'列是否存在，若不存在则创建

## 4. 人工代码审查（逐行中文注释）

# 返回高峰小时的钟点数
peak_hour = hourly_counts.idxmax()

# 获取高峰小时对应的刷卡量数值
peak_hour_count = hourly_counts.max()

# 令时间格式为2位数字，输出
print(f"高峰小时：{peak_hour:02d}:00-{peak_hour+1:02d}:00，刷卡量：{peak_hour_count} 次")

# 从筛选后的数据（df_type0，刷卡类型为0的记录）中提取高峰小时的所有记录
# df_type0['hour'] == peak_hour 创建布尔索引，筛选出该小时的数据
# .copy() 创建独立副本，避免后续操作影响原始数据
peak_hour_data = df_type0[df_type0['hour'] == peak_hour].copy()

# 如果'minute'列不存在，则从交易时间中提取分钟数并创建该列
if 'minute' not in peak_hour_data.columns:
    # 提取分钟部分
    peak_hour_data['minute'] = peak_hour_data['交易时间'].dt.minute

# 定义函数：将分钟数转换为所属5分钟时间段的起始分钟
def get_5min_interval(minute):
    """返回5分钟时间段的起始分钟"""
    # 利用整数除法将分钟数分组，再乘以5得到时间段的起始分钟
    return (minute // 5) * 5

# 将get_5min_interval函数应用到'minute'列的每个值上
peak_hour_data['interval_5min'] = peak_hour_data['minute'].apply(get_5min_interval)

# 按5分钟时间段分组，统计每个时间段的刷卡记录数量
# groupby().size() 返回每个分组的行数（即刷卡次数）
counts_5min = peak_hour_data.groupby('interval_5min').size()

# 返回刷卡量最大时间段的起始分钟
max_5min_start = counts_5min.idxmax()
# 返回最大刷卡量
max_5min_count = counts_5min.max()
# 计算结束分钟
max_5min_end = max_5min_start + 5

# 计算PHF5
# 如果最大5分钟刷卡量为0，则 PHF5 = 0
PHF5 = peak_hour_count / (12 * max_5min_count) if max_5min_count > 0 else 0

# 如果结束分钟≥60，则小时数+1，分钟数-60
print(f"最大5分钟刷卡量（{peak_hour:02d}:{max_5min_start:02d}~{peak_hour+1 if max_5min_end >= 60 else peak_hour:02d}:{max_5min_end if max_5min_end < 60 else max_5min_end-60:02d}）：{max_5min_count} 次")

# 打印PHF5计算公式和结果，保留4位小数
print(f"PHF5 = {peak_hour_count} / (12 × {max_5min_count}) = {PHF5:.4f}")
