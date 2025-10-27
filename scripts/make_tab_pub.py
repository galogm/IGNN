from the_utils import csv_to_table

ROW_ORDER = [
    "r-IGNN",
    "a-IGNN",
    "c-IGNN",
    # "r-IGNN-public",
    # "a-IGNN-public",
    # "c-IGNN-public",
]
COL_ORDER = [
    # "actor",
    # "roman-empire",
    # "squirrel",
    # "chameleon",
    # "amazon-ratings",
    # "pubmed",
    # "wikics",
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
    raw_path="results/public.csv",
    # save_path="results/table_pub.csv",
    save_path="results/table_our.csv",
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
