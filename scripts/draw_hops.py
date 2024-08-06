import ast

import matplotlib.pyplot as plt
import pandas as pd
from the_utils import draw_chart

# Load the data from the CSV file
csv_file = "results/results_v88.1.csv"
data = pd.read_csv(csv_file)


# Function to extract 'n_hops' and 'nrl' from the 'args' column
def extract_args(args_str):
    args_dict = ast.literal_eval(args_str)
    return args_dict.get("n_hops", ""), args_dict.get("nrl", ""), args_dict.get("nie", "")


# Apply the function to extract 'n_hops' and 'nrl'
data["n_hops"], data["nrl"], data["nie"] = zip(*data["args"].apply(extract_args))
data["acc_hl"] = data["acc_hl"].str.split("±").str[0].astype(float)

xticks = [0, 1, 2, 4, 8, 16, 32, 64]
markers = ["*", "^", "o", "p", "s", "d", "v", "<", ">", "h", "H", "x", "D"]

# Plot the data
unique_nrl = data["nrl"].unique()

for nrl in unique_nrl:
    plt.rcParams["lines.markersize"] = 5
    plt.locator_params(axis="y", nbins=10)

    handles = []
    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(
        3, 2, figsize=(10, 4), gridspec_kw={
            "height_ratios": [20, 35, 2],
            "width_ratios": [4, 6]
        }
    )

    subset_nrl = data[data["nie"] == "gcn-nie-nst"].sort_values(by="n_hops")
    subset_nrl = subset_nrl[subset_nrl["nrl"] == nrl]
    cnt = 0
    for dataset in subset_nrl["dataset"].unique():
        # if dataset in ["wikics", "photo", "pubmed", "blogcatalog", "flickr", "roman-empire"]:
        #     continue
        subset = subset_nrl[subset_nrl["dataset"] == dataset]
        (l1, ) = ax1.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
        )
        (l2, ) = ax2.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
            label=dataset,
        )
        (l3, ) = ax3.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
        )
        (l4, ) = ax4.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
        )
        (l5, ) = ax5.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
        )
        (l6, ) = ax6.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
        )
        cnt = cnt + 1

        handles.append(l2)

    y1 = (80, 100)
    y2 = (30, 65)
    y3 = (0, 5)

    x1 = (0, 9)
    x2 = (15, 65)
    # Break the y-axis
    ax1.set_ylim(*y1)  # Top left part
    ax1.set_xlim(*x1)  # Top left part

    ax2.set_ylim(*y2)  # Middle left part
    ax2.set_xlim(*x1)  # Middle left part

    ax3.set_xlim(*x1)  # Bottom left part
    ax3.set_ylim(*y3)  # Bottom left part

    ax4.set_xlim(*x2)  # Top right part
    ax4.set_ylim(*y1)  # Top right part

    ax5.set_xlim(*x2)  # Middle right part
    ax5.set_ylim(*y2)  # Middle right part

    ax6.set_xlim(*x2)  # Bottom right part
    ax6.set_ylim(*y3)  # Bottom right part
    # Add labels and title

    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax4.set_xticks([])
    ax5.set_xticks([])

    ax3.set_xticks([0, 1, 2, 4, 8])
    ax3.set_xticklabels(ax3.get_xticks(), fontsize=20)

    ax6.set_xticks([16, 32, 64])
    ax6.set_xticklabels(ax6.get_xticks(), fontsize=20)

    # ytick1 = ax1.get_yticks().astype(int).tolist()
    # ytick2 = ax2.get_yticks().astype(int).tolist()
    # ytick3 = ax3.get_yticks().astype(int).tolist()
    for ax in [ax1]:
        ax.set_yticks([80, 90, 100])
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)

    for ax in [ax2]:
        ax.set_yticks([30, 40, 50, 60])
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)

    for ax in [ax3]:
        ax.set_yticks([0])
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)

    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    #     ax.set_xscale('function', functions=(lambda x: x**(7/8), lambda x: x**(8/7)))

    labels = subset_nrl["dataset"].unique()
    fig.legend(
        handles,
        labels,
        fontsize=20,
        loc="center left",
        bbox_to_anchor=[0.66, 0.5],
        frameon=False,
    )
    fig.supxlabel("n_hops", fontsize=20)
    fig.supylabel("Accuracy", fontsize=20)

    # plt.xlim(xmin=0,xmax=128)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=0.66)
    # Show the plot
    plt.savefig(f"results/hops_{nrl}.pdf")

for nrl in ["concat"]:

    plt.rcParams["lines.markersize"] = 5
    plt.locator_params(axis="y", nbins=10)

    handles = []
    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(
        3, 2, figsize=(10, 4), gridspec_kw={
            "height_ratios": [20, 35, 2],
            "width_ratios": [4, 6]
        }
    )

    subset_nrl = data[data["nie"] != "gcn-nie-nst"].sort_values(by="n_hops")
    subset_nrl = subset_nrl[subset_nrl["nrl"] == nrl]
    cnt = 0
    for dataset in subset_nrl["dataset"].unique():
        # if dataset in ["wikics", "photo", "pubmed", "blogcatalog", "flickr", "roman-empire"]:
        #     continue
        subset = subset_nrl[subset_nrl["dataset"] == dataset]
        (l1, ) = ax1.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
        )
        (l2, ) = ax2.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
            label=dataset,
        )
        (l3, ) = ax3.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
        )
        (l4, ) = ax4.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
        )
        (l5, ) = ax5.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
        )
        (l6, ) = ax6.plot(
            subset["n_hops"].astype(int),
            subset["acc_hl"],
            marker=markers[cnt],
        )
        cnt = cnt + 1

        handles.append(l2)

    y1 = (80, 100)
    y2 = (30, 65)
    y3 = (0, 5)

    x1 = (0, 9)
    x2 = (15, 65)
    # Break the y-axis
    ax1.set_ylim(*y1)  # Top left part
    ax1.set_xlim(*x1)  # Top left part

    ax2.set_ylim(*y2)  # Middle left part
    ax2.set_xlim(*x1)  # Middle left part

    ax3.set_xlim(*x1)  # Bottom left part
    ax3.set_ylim(*y3)  # Bottom left part

    ax4.set_xlim(*x2)  # Top right part
    ax4.set_ylim(*y1)  # Top right part

    ax5.set_xlim(*x2)  # Middle right part
    ax5.set_ylim(*y2)  # Middle right part

    ax6.set_xlim(*x2)  # Bottom right part
    ax6.set_ylim(*y3)  # Bottom right part
    # Add labels and title

    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax4.set_xticks([])
    ax5.set_xticks([])

    ax3.set_xticks([0, 1, 2, 4, 8])
    ax3.set_xticklabels(ax3.get_xticks(), fontsize=20)

    ax6.set_xticks([16, 32, 64])
    ax6.set_xticklabels(ax6.get_xticks(), fontsize=20)

    # ytick1 = ax1.get_yticks().astype(int).tolist()
    # ytick2 = ax2.get_yticks().astype(int).tolist()
    # ytick3 = ax3.get_yticks().astype(int).tolist()
    for ax in [ax1]:
        ax.set_yticks([80, 90, 100])
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)

    for ax in [ax2]:
        ax.set_yticks([30, 40, 50, 60])
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)

    for ax in [ax3]:
        ax.set_yticks([0])
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)

    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    #     ax.set_xscale('function', functions=(lambda x: x**(7/8), lambda x: x**(8/7)))

    labels = subset_nrl["dataset"].unique()
    fig.legend(
        handles,
        labels,
        fontsize=20,
        loc="center left",
        bbox_to_anchor=[0.66, 0.5],
        frameon=False,
    )
    fig.supxlabel("n_hops", fontsize=20)
    fig.supylabel("Accuracy", fontsize=20)

    # plt.xlim(xmin=0,xmax=128)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1, right=0.66)

    # Show the plot
    plt.savefig(f"results/hops_{nrl}_deepsets.pdf")
