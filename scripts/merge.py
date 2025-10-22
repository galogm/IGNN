import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

font = {"family": "Times New Roman", "weight": "bold", "size": 24}

plt.rc("font", **font)

df = pd.read_csv("results/dml.csv")

df[["Ls_mean", "Ls_std"]] = df["Ls"].str.extract(r"([\d.]+)±([\d.]+)").astype(float)


def fix_array_format(s):
    s = re.sub(r"(\d)\s+(\d)", r"\1, \2", s)
    return np.array(eval(s))


df["H_dms_m"] = df["H_dms_m"].apply(fix_array_format)
df["H_dms_s"] = df["H_dms_s"].apply(fix_array_format)

output_dir = "results/plots_top_bottom"
os.makedirs(output_dir, exist_ok=True)

selected_hops = [10, 32, 64]

for dataset, group in df.groupby("dataset"):
    for hop in selected_hops:
        max_hop = hop
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 15), sharex=False, gridspec_kw={"height_ratios": [2, 3]}
        )

        ax_top = axes[0]
        for model, model_group in group.groupby("model"):
            model_group = model_group[model_group["hops"] <= max_hop]
            if model_group.empty:
                continue

            hops = model_group["hops"]
            ls_mean = model_group["Ls_mean"]
            ls_std = model_group["Ls_std"]

            ax_top.plot(hops, ls_mean, marker="o", label=model)
            ax_top.fill_between(hops, ls_mean - ls_std, ls_mean + ls_std, alpha=0.2)

        ax_top.set_yscale("log")
        ax_top.set_xlabel("Layers/Hops k")
        ax_top.set_ylabel("Lipchitz Constant")
        ax_top.set_title(f"{dataset}")
        ax_top.legend()
        ax_top.grid(True, which="both", linestyle="--", linewidth=0.5)

        ax_bottom = axes[1]
        for model, model_group in group.groupby("model"):
            model_group = model_group[model_group["hops"] == hop]
            if model_group.empty:
                continue

            h_dms_m = model_group["H_dms_m"].values[0]
            h_dms_s = model_group["H_dms_s"].values[0]
            dims = np.arange(len(h_dms_m))

            h_dms_m[0] = np.nan
            h_dms_s[0] = np.nan

            ax_bottom.plot(dims, h_dms_m, marker="o", label=model)
            ax_bottom.fill_between(dims, h_dms_m - h_dms_s, h_dms_m + h_dms_s, alpha=0.2)

        ax_bottom.set_xlabel("Layers/Hops k")
        ax_bottom.set_ylabel(f"D_M(H^k)")
        ax_bottom.set_title(f"{dataset}")
        ax_bottom.legend()
        ax_bottom.grid(True, which="both", linestyle="--", linewidth=0.5)

        ax_bottom.set_yscale("log")
        ax_bottom.set_ylim(0)

        output_path = os.path.join(output_dir, f"{dataset}_hops{hop}.png")
        plt.savefig(output_path)
        output_path = os.path.join(output_dir, f"{dataset}_hops{hop}.pdf")
        plt.savefig(output_path)
        plt.close()

print(f"Analysis plots are documented in {output_dir}")
