from the_utils import csv_to_table

ROW_ORDER = [
    "FlatGNN-gcn-concat",
    "FlatGNN-gcn-lstm",
    "FlatGNN-gcn-mean",
    "FlatGNN-gcn-sum",
    "FlatGNN-gcn-max",
    "FlatGNN-gcn-multi-con",
]

COL_ORDER = [
    "actor",
    # "blogcatalog",
    "flickr",
    # "roman-empire",
    "squirrel",
    "chameleon",
    # "amazon-ratings",
    "pubmed",
    # 'photo',
    # 'wikics',
]

df = csv_to_table(
    raw_path="results/results_v99.3.csv",
    save_path="results/ablation.csv",
    row_key="model",
    col_key="dataset",
    val_key="acc_hl",
    fillna=0,
    row_order=ROW_ORDER,
    col_order=COL_ORDER,
    average_rank=False,
    bold_max=True,
)
