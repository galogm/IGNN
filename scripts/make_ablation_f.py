from the_utils import csv_to_table

ROW_ORDER = [
    "FlatGNN-gcn-concat",
    "FlatGNN-gcn-multi-con",
    "FlatGNN-gcn-sum",
    "FlatGNN-gcn-sum-fc",
    "FlatGNN-gcn-max",
    "FlatGNN-gcn-max-fc",
    "FlatGNN-gcn-mean",
    "FlatGNN-gcn-mean-fc",
    "FlatGNN-gcn-lstm-nfc",
    "FlatGNN-gcn-lstm-fc",
]
COL_ORDER = [
    "actor",
    "blogcatalog",
    "flickr",
    "roman-empire",
    "squirrel",
    "chameleon",
    "amazon-ratings",
    "pubmed",
    "photo",
    "wikics",
]

df = csv_to_table(
    raw_path="results/results_v99.0.csv",
    save_path="results/ablation-f.csv",
    row_key="model",
    col_key="dataset",
    val_key="acc_hl",
    fillna=0,
    row_order=ROW_ORDER,
    col_order=COL_ORDER,
    average_rank=False,
    bold_max=True,
)
