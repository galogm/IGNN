# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
import copy
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score as ACC
from sklearn.preprocessing import normalize
from the_utils import get_str_time, make_parent_dirs, save_to_csv_files
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import LayerNorm, Linear, Module, ModuleList
from torch.utils.tensorboard import SummaryWriter
from torch_sparse import SparseTensor, fill_diag
from tqdm import tqdm

from semi_heter.utils.common import sparse_mx_to_torch_sparse_tensor

from ..modules import MLP, InnerProductDecoder, LinTrans, SampleDecoder
from ..utils import preprocess_graph
from .experts import (
    AdaptiveLearning,
    EdgeReconstruction,
    NeighborhoodPrediction,
    SelfReconstruction,
    update_similarity,
    update_threshold,
)


class Deepsets(nn.Module):
    def __init__(self, in_feats, h_feats, dropout=0.5, mean=False):
        super().__init__()
        self.dropout = dropout
        layers = 1
        hs = [h_feats]
        # n = int(math.log2(in_feats)) - math.log2(h_feats)
        # while n // 2 > 0:
        #     layers += 1
        #     n = n // 2
        #     hs.append(hs[-1] * 2)
        self.phi = MLP(
            in_feats=in_feats,
            h_feats=hs[::-1],
            layers=layers,
            acts=[nn.ReLU()] * layers,
            dropout=self.dropout,
        )
        # self.phi = MLP(
        #     in_feats=in_feats,
        #     h_feats=[h_feats],
        #     layers=1,
        #     acts=[nn.ReLU()],
        # )
        self.rho = MLP(
            in_feats=h_feats,
            h_feats=[h_feats],
            layers=1,
            acts=[nn.ReLU()],
            dropout=self.dropout,
        )

    def forward(self, x, mean=False):
        if mean:
            h = self.phi(x)
            return self.rho(torch.mean(h, dim=1))
        h = self.phi(x)
        return h
        # return self.rho(
        #     F.dropout(
        #         h,
        #         self.dropout,
        #         training=self.training,
        #     )
        # )


def preprocess_neighborhoods(
    adj: SparseTensor,
    features: torch.FloatTensor,
    name: str,
    n_hops: int,
    set_diag=True,
    remove_diag=False,
    symm_norm=False,
    device: torch.device = torch.device("cpu"),
    no_save=False,
    return_adj=False,
    process_adj=True,
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
    base = Path(f"tmp/setgnn/neighborhoods/{name}/{sym}")
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


class SetGNN(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats,
        n_hops,
        # n_relations,
        dropout=0.5,
        n_intervals=3,
        no_save=False,
    ):
        super().__init__()
        self.no_save = no_save
        self.n_hops = n_hops
        self.dropout = dropout
        self.h_feats = h_feats
        self.deepsets = nn.ModuleList(
            Deepsets(
                in_feats=in_feats,
                h_feats=h_feats,
                dropout=self.dropout,
            )
            for _ in range(self.n_hops + 1)
        )
        self.readout = Deepsets(
            in_feats=h_feats,
            h_feats=h_feats,
            dropout=self.dropout,
        )
        self.interval = n_intervals
        self.nei_feats = None
        self.n_relations = self.n_hops + 1 - self.interval
        self.relation_nets = nn.ModuleList(
            [
                *[
                    MLP(
                        in_feats=h_feats * self.interval,
                        h_feats=[h_feats],
                        layers=1,
                        acts=[nn.ReLU()],
                        dropout=self.dropout,
                    )
                    for _ in range(self.n_relations + 1)
                ],
            ]
        )
        self.adj = None

    def forward(
        self,
        graph: dgl.DGLGraph,
        device: torch.device,
        feats=None,
    ):
        self.to(device=device)
        self.ns = []
        adj = graph.adj()
        features = graph.ndata["feat"] if feats is None else feats
        if self.nei_feats is None and self.no_save == False:
            self.nei_feats = preprocess_neighborhoods(
                adj=SparseTensor(
                    row=adj.row.long(),
                    col=adj.col.long(),
                    sparse_sizes=adj.shape,
                ),
                features=features,
                name=graph.name,
                n_hops=self.n_hops,
                set_diag=True,
                remove_diag=False,
                symm_norm=True,
                device=device,
            )
        elif self.no_save == True:
            self.nei_feats, self.adj = preprocess_neighborhoods(
                adj=(
                    SparseTensor(
                        row=adj.row.long(),
                        col=adj.col.long(),
                        sparse_sizes=adj.shape,
                    )
                    if self.adj is None
                    else self.adj
                ),
                features=features,
                name=graph.name,
                n_hops=self.n_hops,
                set_diag=True,
                remove_diag=False,
                symm_norm=True,
                device=device,
                no_save=True,
                return_adj=True,
                process_adj=False if self.adj is not None else True,
            )

        for i in range(self.n_hops + 1):
            self.ns.append(self.deepsets[i](self.nei_feats[i]))
        # return [self.readout(torch.stack(self.ns, dim=1), mean=True)]
        hops_feats = torch.cat(self.ns, dim=1)
        # return [hops_feats]
        return [
            torch.cat(
                [
                    self.relation_nets[i](
                        hops_feats[:, self.h_feats * i : self.h_feats * (i + self.interval)]
                    )
                    for i in range(self.n_relations + 1)
                ],
                dim=1,
            )
        ]
        # return [self.mlp(F.dropout(torch.cat(self.ns, dim=1), self.dropout, self.training))]
