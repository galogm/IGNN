from the_utils import csv_to_table

ROW_ORDER = [
    "GCN",
    "SIGN w/o SN",
    "JKNet-GCN",
    "SIGN",
    "r-IGNN",
    "c-IGNN",
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
    raw_path="results/abl.csv",
    save_path="results/table_abl.csv",
    row_key="model",
    col_key="dataset",
    val_key="acc",
    fillna=0,
    row_order=ROW_ORDER,
    col_order=COL_ORDER,
    average_rank=True,
    bold_max=True,
    save_latex=True,
)
