import pandas as pd

# 读取 CSV
df = pd.read_csv('results/times-a.csv')

# 只保留需要的列
df = df[['model', 'hops', 'mean']]

# 只保留 hops ≤ 32 的数据
df = df[df['hops'] <= 32]

# 为排名使用原始数值列（不放大）
df['mean_numeric'] = df['mean'].apply(lambda x: float(x.split('±')[0]) if isinstance(x, str) and '±' in x else float(x))

# 计算平均排名
rank_df = df.pivot(index='model', columns='hops', values='mean_numeric')
avg_rank = rank_df.rank(axis=0, method='average').mean(axis=1)
rank_df['avg_rank'] = avg_rank

# 将 mean 列放大
def amplify_mean_std(x):
    if isinstance(x, str) and '±' in x:
        mean, std = x.split('±')
        return f"{float(mean)*10:.2f}±{float(std)*10:.2f}"
    try:
        return f"{float(x)*10:.2f}"
    except:
        return x

df['mean'] = df['mean'].apply(amplify_mean_std)

# 重新构建放大的透视表
pivot_df = df.pivot(index='model', columns='hops', values='mean')
pivot_df = pivot_df[sorted(pivot_df.columns)]

# 添加 avg_rank 列（保留两位小数）
pivot_df['avg_rank'] = rank_df['avg_rank'].apply(lambda x: f"{x:.2f}")

# 构建 markdown 表格
header = "| Model | " + " | ".join(map(str, pivot_df.columns)) + " |\n"
separator = "|---" + "|---" * len(pivot_df.columns) + "|\n"
rows = [f"| {model} | " + " | ".join(pivot_df.loc[model].fillna("").astype(str)) + " |\n" for model in pivot_df.index]

# 合并写入文件
markdown = header + separator + "".join(rows)

with open('results/time-a.md', 'w') as f:
    f.write(markdown)
