"""
common utils
"""

from typing import Tuple, Union, overload

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from ogb.nodeproppred import Evaluator
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import roc_auc_score
from the_utils import onetime_reminder

evaluators = {
    "products_ogb": Evaluator(name="ogbn-products"),
    "arxiv_ogb": Evaluator(name="ogbn-arxiv"),
}

FloatLike = Union[float, np.floating, torch.Tensor]


@overload
def metric(name, logits, labels) -> float: ...
@overload
def metric(name, logits, labels, train_mask, val_mask, test_mask) -> Tuple[float, float, float]: ...


def metric(
    name, logits, labels, train_mask=None, val_mask=None, test_mask=None
) -> Union[Tuple[FloatLike, FloatLike, FloatLike], FloatLike]:
    if name in [
        "Penn94_linxk",
        "genius_linkx",
        "twitch-gamers_linkx",
        "pokec_linkx",
        "Penn94_linkx",
    ]:
        if labels.min().item() == -1:
            labels[labels == -1] = 0
        # if name in ["genius_linkx"]:
        #     labels = F.one_hot(labels)
    elif name in ["proteins_ogb"]:
        labels = labels.to(torch.float)

    if name not in (
        "yelp-chi_linkx",
        "deezer-europe_linkx",
        # "twitch-gamers_linkx",
        # "Penn94_linkx",
        "fb100_linkx",
        "proteins_ogb",
        "genius_linkx",
    ):
        y_pred = torch.argmax(logits, dim=1, keepdim=True)

        if name in evaluators:
            onetime_reminder(f"Evaluate {name} with OGB Evaluator.\n")
            if train_mask is None:
                return evaluators[name].eval(
                    {
                        "y_true": labels.unsqueeze(1),
                        "y_pred": y_pred,
                    }
                )["acc"]

            train_acc = evaluators[name].eval(
                {
                    "y_true": labels[train_mask].cpu().unsqueeze(1),
                    "y_pred": y_pred[train_mask].cpu(),
                }
            )["acc"]

            valid_acc = evaluators[name].eval(
                {
                    "y_true": labels[val_mask].cpu().unsqueeze(1),
                    "y_pred": y_pred[val_mask].cpu(),
                }
            )["acc"]

            test_acc = evaluators[name].eval(
                {
                    "y_true": labels[test_mask].cpu().unsqueeze(1),
                    "y_pred": y_pred[test_mask].cpu(),
                }
            )["acc"]
            return train_acc, valid_acc, test_acc

        onetime_reminder(f"Evaluate {name} with ACC.\n")
        if train_mask is None:
            return ACC(labels.cpu(), y_pred.cpu())

        return (
            ACC(labels[train_mask].cpu(), y_pred[train_mask].cpu()),
            ACC(labels[val_mask].cpu(), y_pred[val_mask].cpu()),
            ACC(labels[test_mask].cpu(), y_pred[test_mask].cpu()),
        )

    onetime_reminder(f"Evaluate {name} with ROCAUC.\n")
    if train_mask is None:
        return eval_rocauc(labels.cpu().numpy(), logits.cpu().numpy())["rocauc"]

    return (
        eval_rocauc(
            labels[train_mask].cpu().numpy(),
            logits[train_mask].cpu().numpy(),
        )["rocauc"],
        eval_rocauc(
            labels[val_mask].cpu().numpy(),
            logits[val_mask].cpu().numpy(),
        )["rocauc"],
        eval_rocauc(
            labels[test_mask].cpu().numpy(),
            logits[test_mask].cpu().numpy(),
        )["rocauc"],
    )


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


def graph_diameter(g):
    """
    Computes the diameter of a given graph.

    Parameters:
    g (DGLGraph): The input graph.

    RetuRNs:
    int: The diameter of the graph.
    """
    # Convert the DGL graph to a NetworkX graph
    nx_graph = g.to_networkx()

    # Use NetworkX to compute the all-pairs shortest path lengths
    lengths = dict(nx.all_pairs_shortest_path_length(nx_graph))

    # Find the maximum shortest path length
    max_length = 0
    for src in lengths:
        for dst in lengths[src]:
            if lengths[src][dst] > max_length:
                max_length = lengths[src][dst]

    return max_length
