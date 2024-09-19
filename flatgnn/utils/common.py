"""
common utils
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from the_utils import make_parent_dirs
from the_utils import split_train_test_nodes
from torch_sparse import SparseTensor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def eval_rocauc(y_true, y_pred):
    """
    compute ROC-AUC and AP score averaged across tasks
    """

    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

    return {"rocauc": sum(rocauc_list) / len(rocauc_list)}


def get_splits_mask(
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
                else torch.LongTensor(
                    train_idx,
                )
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
                else torch.LongTensor(
                    val_idx,
                )
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
                else torch.LongTensor(
                    test_idx,
                )
            ),
            1,
        )
        .bool()
    )

    return train_mask, val_mask, test_mask


def row_normalized_adjacency(adj, return_deg=False):
    adj = sp.coo_matrix(adj)
    # adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    adj_normalized = adj / row_sum
    if return_deg:
        return sp.coo_matrix(adj_normalized), row_sum
    return sp.coo_matrix(adj_normalized)


def sys_normalized_adjacency(adj, return_deg=False):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    if return_deg:
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(), rowsum
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse tensor

    Args:
        sparse_mx (<class 'scipy.sparse'>): sparse matrix

    Returns:
        (torch.Tensor): torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def preprocess_neighborhoods(
    adj: SparseTensor,
    features: torch.FloatTensor,
    name: str,
    n_hops: int,
    set_diag=True,
    remove_diag=False,
    symm_norm=False,
    row_normalized=True,
    device: torch.device = torch.device("cpu"),
    no_save=False,
    return_adj=False,
    process_adj=True,
    save_dir: str = "tmp/flatgnn/neighborhoods",
):
    if process_adj:
        if set_diag:
            print("... setting diagonal entries")
            adj = adj.set_diag()
        elif remove_diag:
            print("... removing diagonal entries")
            adj = adj.remove_diag()
        else:
            print("... keeping diag elements as they are")
        if symm_norm:
            print("... performing symmetric normalization")
            deg = adj.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        else:
            print("... performing asymmetric normalization")
            deg = adj.sum(dim=1).to(torch.float)
            deg_inv = deg.pow(-1.0)
            deg_inv[deg_inv == float("inf")] = 0
            adj = deg_inv.view(-1, 1) * adj

    # features = F.normalize(features, dim=1) if row_normalized else features
    nei_feats = [features.to(device)]
    if no_save:
        adj = adj.to_torch_sparse_csr_tensor() if process_adj else adj
        for i in range(1, n_hops + 1):
            x = torch.mm(adj, features.cpu())
            nei_feats.append(x.to(torch.float).to(device))
        if return_adj:
            return nei_feats, adj
        return nei_feats

    adj = adj.to_scipy(layout="csr")
    x = features.numpy()
    sym = "sym" if symm_norm else "nsy"
    diag = "diag" if set_diag else "ndiag"
    norm = "norm" if row_normalized else "nnorm"
    base = Path(f"{save_dir}/{name}/{norm}/{diag}/{sym}")
    for i in tqdm(range(1, n_hops + 1)):
        file = base.joinpath(f"{i}")
        if file.exists():
            x = torch.load(file, map_location=device)
            nei_feats.append(x)
            x = x.cpu().numpy()
            continue
        make_parent_dirs(file)
        x = adj @ x
        nei_feats.append(torch.from_numpy(x).to(torch.float).to(device))
        torch.save(
            obj=nei_feats[-1],
            f=file,
            pickle_protocol=4,
        )
    return nei_feats
