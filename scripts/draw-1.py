import os
import re  # 用于正则表达式处理格式错误

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("results/dml.csv")

# 修正数组格式：在数值之间添加逗号
def fix_array_format(s):
    s = re.sub(r"(\d)\s+(\d)", r"\1, \2", s)  # 在数字之间插入逗号
    return np.array(eval(s))  # 转换为 numpy 数组

# 解析 'H_dms_m' 和 'H_dms_s'
df["H_dms_m"] = df["H_dms_m"].apply(fix_array_format)
df["H_dms_s"] = df["H_dms_s"].apply(fix_array_format)

# 确保输出文件夹存在
output_dir = "results/plots_h_dms"
os.makedirs(output_dir, exist_ok=True)

# 遍历所有数据集和 hops 组合
for (dataset, hops), group in df.groupby(["dataset", "hops"]):
    plt.figure(figsize=(8, 6))

    # 遍历所有模型
    for model, model_group in group.groupby("model"):
        h_dms_m = model_group["H_dms_m"].values[0]  # 取出唯一的数组
        h_dms_s = model_group["H_dms_s"].values[0]  # 取出唯一的数组
        dims = np.arange(len(h_dms_m))  # 横坐标：H_dms_m 的维度索引
        h_dms_m[0]=np.nan
        h_dms_s[0]=np.nan
        # 画均值曲线
        plt.plot(dims, h_dms_m, marker="o", label=model)

        # 画标准差阴影
        plt.fill_between(dims, h_dms_m - h_dms_s, h_dms_m + h_dms_s, alpha=0.2)

    if hops>10:
        # 使用对数刻度防止大数值压缩
        plt.yscale("log")
    else:
        plt.ylim([0, 3500])


    # 图表设置
    plt.xlabel("Layers/Hops k")
    plt.ylabel(f"D_M(H^k) {'' if hops<=10 else '    -    Log Scale'}")
    plt.title(f"{dataset} ({'homophily' if dataset=='cora' else 'heterophily'})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 保存图像
    output_path = os.path.join(output_dir, f"{dataset}_hops{hops}.pdf")
    plt.savefig(output_path)
    plt.close()

print("所有符合条件的图表已生成并保存到 'plots_h_dms' 文件夹。")
