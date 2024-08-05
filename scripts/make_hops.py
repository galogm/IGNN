from the_utils import csv_to_table

ROW_ORDER = [
    "FlatGNN-gcn-nie-nst-1",
    "FlatGNN-gcn-nie-nst-2",
    "FlatGNN-gcn-4",
    "FlatGNN-gcn-8",
    "FlatGNN-gcn-16",
    "FlatGNN-gcn-32",
    "FlatGNN-gcn-nie-nst-64",
]

COL_ORDER = [
    "actor",
    # "blogcatalog",
    # "flickr",
    # "roman-empire",
    "squirrel",
    "chameleon",
    # "amazon-ratings",
    "pubmed",
    # 'photo',
    # 'wikics',
]

df = csv_to_table(
    raw_path="results/results_v99.8.csv",
    save_path="results/hops.csv",
    row_key="model",
    col_key="dataset",
    val_key="acc_hl",
    fillna=0,
    row_order=ROW_ORDER,
    col_order=COL_ORDER,
    average_rank=False,
    bold_max=False,
)
