"""DATA"""

# pylint: disable=invalid-name,too-few-public-methods
from dataclasses import dataclass


@dataclass
class DataConf:
    """Data config"""

    DATA_DIR: str
    SPLIT_DIR: str


@dataclass
class INConf:
    """Inceptive aggregation configs.

    Args:
        name (str): The name of the dataset.
        n_hops (int): The number of hops for aggregation.
        add_self_loop (bool, optional): Whether to add self-loops. Defaults to True.
        remove_self_loop (bool, optional): Whether to remove self-loops. Defaults to False.
        symm_norm (bool, optional): Whether to use symmetric normalization; \
            if False, uses asymmetric normalization. Defaults to False.
        row_normalized (bool, optional): Whether to perform row normalization. Defaults to False.
        fast (bool, optional):  Whether to use the fast c-IGNN. Defaults to False.
        save_dir (str, optional): Directory path for caching. Defaults to "tmp/ignn/neighborhoods".
    """

    name: str
    n_hops: int
    add_self_loop: bool = True
    remove_self_loop: bool = False
    symm_norm: bool = False
    row_normalized: bool = False
    fast: bool = False
    save_dir: str = "tmp/ignn/neighborhoods"
