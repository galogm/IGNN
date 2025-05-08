import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 24}

plt.rc('font', **font)

# 读取 CSV 文件
df = pd.read_csv("results/dml.csv")

# 处理 'Ls' 列中的均值和标准差
df[['Ls_mean', 'Ls_std']] = df['Ls'].str.extract(r'([\d.]+)±([\d.]+)').astype(float)

# 修正数组格式：在数值之间添加逗号
def fix_array_format(s):
    s = re.sub(r"(\d)\s+(\d)", r"\1, \2", s)  # 在数字之间插入逗号
    return np.array(eval(s))  # 转换为 numpy 数组

# 解析 'H_dms_m' 和 'H_dms_s'
df["H_dms_m"] = df["H_dms_m"].apply(fix_array_format)
df["H_dms_s"] = df["H_dms_s"].apply(fix_array_format)

# 确保输出文件夹存在
output_dir = "plots_top_bottom"
os.makedirs(output_dir, exist_ok=True)

# 指定要绘制 D_M(X) 的 hops 值
selected_hops = [10, 32, 64]
# max_hop = max(selected_hops)  # 最大 hops 值

# 遍历所有数据集
for dataset, group in df.groupby("dataset"):
    for hop in selected_hops:
        max_hop=hop
        fig, axes = plt.subplots(2, 1, figsize=(10, 15), sharex=False, gridspec_kw={'height_ratios': [2, 3]})

        # 上方：Lipchitz Constant（覆盖所有 hops < max_hop）
        ax_top = axes[0]
        for model, model_group in group.groupby("model"):
            # 选取所有 hops < max_hop
            model_group = model_group[model_group["hops"] <= max_hop]
            if model_group.empty:
                continue  # 跳过没有数据的情况

            hops = model_group["hops"]
            ls_mean = model_group["Ls_mean"]
            ls_std = model_group["Ls_std"]

            ax_top.plot(hops, ls_mean, marker="o", label=model)
            ax_top.fill_between(hops, ls_mean - ls_std, ls_mean + ls_std, alpha=0.2)

        ax_top.set_yscale("log")  # 纵轴对数缩放
        ax_top.set_xlabel("Layers/Hops k")
        ax_top.set_ylabel("Lipchitz Constant     -     Log scale")
        ax_top.set_title(f"{dataset}")
        ax_top.legend()
        ax_top.grid(True, which="both", linestyle="--", linewidth=0.5)

        # 下方：D_M(X)（仅绘制 hops=10, 32, 64）
        ax_bottom = axes[1]
        for model, model_group in group.groupby("model"):
            model_group = model_group[model_group["hops"] == hop]
            if model_group.empty:
                continue

            h_dms_m = model_group["H_dms_m"].values[0]
            h_dms_s = model_group["H_dms_s"].values[0]
            dims = np.arange(len(h_dms_m))

            # 忽略第一个点
            h_dms_m[0] = np.nan
            h_dms_s[0] = np.nan

            ax_bottom.plot(dims, h_dms_m, marker="o", label=model)
            ax_bottom.fill_between(dims, h_dms_m - h_dms_s, h_dms_m + h_dms_s, alpha=0.2)

        ax_bottom.set_xlabel("Layers/Hops k")
        ax_bottom.set_ylabel(f"D_M(H^k) {'' if hop<=10 else '     -     Log Scale'}")
        ax_bottom.set_title(f"{dataset}")
        ax_bottom.legend()
        ax_bottom.grid(True, which="both", linestyle="--", linewidth=0.5)

        # 仅 hops > 10 使用对数刻度
        if hop > 10:
            ax_bottom.set_yscale("log")
            ax_bottom.set_ylim(0)
        else:
            ax_bottom.set_ylim([0, 3500])

        # 保存图像
        output_path = os.path.join(output_dir, f"{dataset}_hops{hop}.png")
        plt.savefig(output_path)
        output_path = os.path.join(output_dir, f"{dataset}_hops{hop}.pdf")
        plt.savefig(output_path)
        plt.close()

print("所有符合条件的图表已生成并保存到 'plots_top_bottom' 文件夹。")
