import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("results/dml.csv")

# 处理 'Ls' 列中的均值和标准差
df[['Ls_mean', 'Ls_std']] = df['Ls'].str.extract(r'([\d.]+)±([\d.]+)').astype(float)

# 确保输出文件夹存在
output_dir = "results/plots"
os.makedirs(output_dir, exist_ok=True)

Hop = 16
# 遍历所有数据集
for dataset, group in df.groupby("dataset"):
    plt.figure(figsize=(8, 6))

    # 遍历所有模型
    for model, model_group in group.groupby("model"):
        # 仅保留 hops <= 32 的数据
        model_group = model_group[model_group["hops"] <= Hop][model_group["hops"] >=1]

        if model_group.empty:
            continue  # 如果数据为空，跳过

        hops = model_group["hops"]
        ls_mean = model_group["Ls_mean"]
        ls_std = model_group["Ls_std"]

        # 画均值曲线
        plt.plot(hops, ls_mean, marker="o", label=model)

        # 画标准差阴影
        plt.fill_between(hops, ls_mean - ls_std, ls_mean + ls_std, alpha=0.2)

    # 使用对数刻度防止大数值压缩
    plt.yscale("log")

    # 图表设置
    plt.xlabel("Hops")
    plt.ylabel("Lipchitz Constant     -     Log scale)")
    plt.title(f"{dataset} ({'homophily' if dataset=='cora' else 'heterophily'})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 保存图像
    output_path = os.path.join(output_dir, f"{dataset}-{Hop}-L.pdf")
    plt.savefig(output_path)
    plt.close()

print("所有符合条件 (hops <= 32) 的图表已生成并保存到 'plots' 文件夹。")
