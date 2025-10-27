"""Get Splits.
"""

# pylint: disable=invalid-name,
from typing import Tuple

import numpy as np
import torch
from the_utils import split_train_test_nodes
from torch_geometric.utils import index_to_mask

from ignn.utils.common import onetime_reminder


def get_random_split_masks(
    name,
    n_nodes,
    train_ratio,
    valid_ratio,
    repeat,
    split_id,
    SPLIT_DIR,
    labeled_idx=None,
):
    n_nodes = n_nodes if labeled_idx is None else len(labeled_idx)
    train_idx, val_idx, test_idx = split_train_test_nodes(
        num_nodes=n_nodes,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        data_name=name,
        split_id=split_id,
        split_times=repeat,
        fixed_split=True,
        split_save_dir=SPLIT_DIR,
    )
    if labeled_idx is not None:
        train_idx = labeled_idx[train_idx]
        val_idx = labeled_idx[val_idx]
        test_idx = labeled_idx[test_idx]

    train_mask = index_to_mask(torch.from_numpy(train_idx), n_nodes)
    val_mask = index_to_mask(torch.from_numpy(val_idx), n_nodes)
    test_mask = index_to_mask(torch.from_numpy(test_idx), n_nodes)

    return train_mask, val_mask, test_mask


def get_splits(
    data,
    name,
    num_nodes,
    i,
    repeat=10,
    TRAIN_RATIO=None,
    VALID_RATIO=None,
    DATA=None,
    labeled_idx=None,
    public=False,
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
    train_mask = data.get("train_mask", None)
    val_mask = data.get("val_mask", None)
    test_mask = data.get("test_mask", None)
    if train_mask is not None and public:
        if "ogb" in name or "pubmed" in name or "cora" in name:
            num = 1
        else:
            num = train_mask.shape[1]
            train_mask = train_mask[:, i] if len(train_mask.shape) > 1 else train_mask
            val_mask = val_mask[:, i] if len(val_mask.shape) > 1 else val_mask
            test_mask = test_mask[:, i] if len(test_mask.shape) > 1 else test_mask
        onetime_reminder(
            f"split num: {num}; "
            f"public split train:val:test = {train_mask.sum()/ num_nodes*100:.2f}:"
            f"{val_mask.sum()/ num_nodes*100:.2f}:"
            f"{test_mask.sum()/ num_nodes*100:.2f}"
        )
    elif name == "pokec_linkx":
        split = np.load("data/random_splits/fixed_splits/pokec-splits.npy", allow_pickle=True)

        train_mask = index_to_mask(torch.from_numpy(np.asarray(split[i]["train"])), num_nodes)
        val_mask = index_to_mask(torch.from_numpy(np.asarray(split[i]["valid"])), num_nodes)
        test_mask = index_to_mask(torch.from_numpy(np.asarray(split[i]["test"])), num_nodes)
        onetime_reminder(
            f"split num: {split.shape}\n"
            f"public splits train:val:test = {train_mask.sum()/ num_nodes*100:.0f}:"
            f"{val_mask.sum()/ num_nodes*100:.0f}:"
            f"{test_mask.sum()/ num_nodes*100:.0f}\n"
        )
    else:
        train_mask, val_mask, test_mask = get_random_split_masks(
            name=name,
            n_nodes=num_nodes,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            repeat=repeat,
            split_id=i,
            SPLIT_DIR=DATA.SPLIT_DIR,
            labeled_idx=(labeled_idx if name in ["wiki_linkx", "pokec_linkx"] else None),
        )
        onetime_reminder(
            f"split num: {repeat}\n"
            "random splits train:val:test ="
            f"{TRAIN_RATIO}:{VALID_RATIO}:{100-TRAIN_RATIO-VALID_RATIO}\n"
        )

    return train_mask, val_mask, test_mask
