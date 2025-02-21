"""Get Splits.
"""

from typing import Tuple

import numpy as np
import torch
from the_utils import split_train_test_nodes

from .common import onetime_reminder


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
    if labeled_idx is not None:
        train_idx, val_idx, test_idx = split_train_test_nodes(
            num_nodes=len(labeled_idx),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            data_name=name,
            split_id=split_id,
            split_times=repeat,
            fixed_split=True,
            split_save_dir=SPLIT_DIR,
        )
        train_idx = labeled_idx[train_idx]
        val_idx = labeled_idx[val_idx]
        test_idx = labeled_idx[test_idx]
    else:
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

    train_mask = (
        torch.zeros(n_nodes)
        .scatter_(
            0,
            (
                train_idx.to(torch.int64)
                if isinstance(train_idx, torch.Tensor)
                else torch.LongTensor(train_idx)
            ),
            1,
        )
        .bool()
    )
    val_mask = (
        torch.zeros(n_nodes)
        .scatter_(
            0,
            (
                val_idx.to(torch.int64)
                if isinstance(val_idx, torch.Tensor)
                else torch.LongTensor(val_idx)
            ),
            1,
        )
        .bool()
    )
    test_mask = (
        torch.zeros(n_nodes)
        .scatter_(
            0,
            (
                test_idx.to(torch.int64)
                if isinstance(test_idx, torch.Tensor)
                else torch.LongTensor(test_idx)
            ),
            1,
        )
        .bool()
    )

    return train_mask, val_mask, test_mask


def get_splits(
    graph,
    label,
    i,
    repeat=10,
    TRAIN_RATIO=None,
    VALID_RATIO=None,
    DATA=None,
    labeled_idx=None,
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
    if "train_mask" in graph.ndata and graph.name not in [
        "cora_pyg",
        "actor_pyg",
        "pubmed_pyg",
        "wikics_pyg",
        "arxiv-year_linkx",
    ]:
        train_mask, val_mask, test_mask = (
            graph.ndata["train_mask"],
            graph.ndata["val_mask"],
            graph.ndata["test_mask"],
        )
        onetime_reminder(
            f"split num: {1}\n"
            f"public split train:val:test = {train_mask.sum()/ graph.num_nodes()*100:.0f}:"
            f"{val_mask.sum()/ graph.num_nodes()*100:.0f}:"
            f"{test_mask.sum()/ graph.num_nodes()*100:.0f}\n"
        )
    elif graph.name == "pokec_linkx":
        split = np.load(f"data/random_splits/fixed_splits/pokec-splits.npy", allow_pickle=True)
        train_idx = torch.from_numpy(np.asarray(split[i]["train"]))
        val_idx = torch.from_numpy(np.asarray(split[i]["valid"]))
        test_idx = torch.from_numpy(np.asarray(split[i]["test"]))

        train_idx = train_idx[label[train_idx] != -1]
        val_idx = val_idx[label[val_idx] != -1]
        test_idx = test_idx[label[test_idx] != -1]

        train_mask = (
            torch.zeros(graph.num_nodes())
            .scatter_(
                0,
                (
                    train_idx.to(torch.int64)
                    if isinstance(train_idx, torch.Tensor)
                    else torch.LongTensor(train_idx)
                ),
                1,
            )
            .bool()
        )
        val_mask = (
            torch.zeros(graph.num_nodes())
            .scatter_(
                0,
                (
                    val_idx.to(torch.int64)
                    if isinstance(val_idx, torch.Tensor)
                    else torch.LongTensor(val_idx)
                ),
                1,
            )
            .bool()
        )
        test_mask = (
            torch.zeros(graph.num_nodes())
            .scatter_(
                0,
                (
                    test_idx.to(torch.int64)
                    if isinstance(test_idx, torch.Tensor)
                    else torch.LongTensor(test_idx)
                ),
                1,
            )
            .bool()
        )
        onetime_reminder(
            f"split num: {split.shape}\n"
            f"public splits train:val:test = {train_mask.sum()/ graph.num_nodes()*100:.0f}:"
            f"{val_mask.sum()/ graph.num_nodes()*100:.0f}:"
            f"{test_mask.sum()/ graph.num_nodes()*100:.0f}\n"
        )
    else:
        train_mask, val_mask, test_mask = get_random_split_masks(
            name=graph.name,
            n_nodes=graph.num_nodes(),
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            repeat=repeat,
            split_id=i,
            SPLIT_DIR=DATA.SPLIT_DIR,
            labeled_idx=(labeled_idx if graph.name in ["wiki_linkx", "pokec_linkx"] else None),
        )
        onetime_reminder(
            f"split num: {repeat}\n"
            f"random splits train:val:test = {TRAIN_RATIO}:{VALID_RATIO}:{100-TRAIN_RATIO-VALID_RATIO}\n"
        )

    return train_mask, val_mask, test_mask
