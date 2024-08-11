from the_utils import csv_to_table

ROW_ORDER = [
    "FlatGNN-gcn-nie-nst-only-concat",
    "FlatGNN-gcn-nie-nst-concat",
    "FlatGNN-gcn-nie-nst-lstm",
    "FlatGNN-gcn-nie-nst-mean",
    "FlatGNN-gcn-nie-nst-sum",
    "FlatGNN-gcn-nie-nst-max",
    # "FlatGNN-gcn-multi-con",
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
    raw_path="results/nrl.csv",
    save_path="results/nrl-f.csv",
    row_key="model",
    col_key="dataset",
    val_key="acc_hl",
    fillna=0,
    row_order=ROW_ORDER,
    col_order=COL_ORDER,
    average_rank=False,
    bold_max=True,
)
