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
        hs = [h_feats // 2] if not mean else [h_feats]
        # n = int(math.log2(in_feats)) - math.log2(h_feats)
        # while n // 4 > 0:
        #     layers += 1
        #     n = n // 4
        #     hs.append(hs[-1] * 4)
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

    def forward(self, x, adj=None, mean=False):
        if adj is not None:
            h = self.phi(x)
            return self.rho(
                F.dropout(
                    torch.cat(
                        [
                            torch.mm(adj, h),
                            h,
                        ],
                        dim=1,
                    ),
                    self.dropout,
                    training=self.training,
                )
            )
        if mean:
            h = self.phi(x)
            return self.rho(
                F.dropout(
                    torch.mean(h, dim=1),
                    self.dropout,
                    training=self.training,
                )
            )

        h = self.phi(x)
        return self.rho(
            F.dropout(
                torch.cat(
                    [
                        h,
                        h,
                    ],
                    dim=1,
                ),
                self.dropout,
                training=self.training,
            )
        )


class SetGNN(nn.Module):
    def __init__(self, in_feats, h_feats, n_hops, dropout=0.5):
        super().__init__()
        self.n_hops = n_hops
        self.dropout = dropout
        # layers = 1
        # hs = [h_feats]
        # n = int(math.log2(in_feats)) - int(math.log2(h_feats))
        # while n // 2 > 0:
        #     layers += 1
        #     n = n // 2
        #     hs.append(hs[-1] * 2)
        # self.mlp = MLP(
        #     in_feats=h_feats * self.n_hops,
        #     h_feats=hs[::-1],
        #     layers=layers,
        #     acts=[nn.ReLU()] * layers,
        #     dropout=self.dropout,
        # )
        self.deepsets = nn.ParameterList(
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
        self.adjs = None
        self.ns = None

    def forward(self, x, graph, device, uniform=False, k_kernel=0):
        # ego_feats = self.mlp(x)
        self.ns = []
        i = (
            8
            if self.n_hops <= 8
            and Path(f"tmp/setgnn/{graph.name}-8-u{uniform}-k{k_kernel}").exists()
            else self.n_hops
        )
        tmp = Path(f"tmp/setgnn/{graph.name}-{i}-u{uniform}-k{k_kernel}")

        if k_kernel != 0:
            graph = dgl.knn_graph(x, k_kernel)

        LOAD = True
        if self.adjs is None:
            if LOAD and tmp.exists():
                self.adjs = torch.load(tmp, map_location=device)
            else:
                make_parent_dirs(tmp)
                adj_csr = adj_csr_raw = (
                    graph.remove_self_loop().add_self_loop().adj_external(scipy_fmt="csr")
                ).astype(float)

                self.adjs = [torch.eye(graph.num_nodes()).to(device)]
                for i in range(self.n_hops):
                    if i > 0:
                        adj_csr = adj_csr_raw = adj_csr_raw.dot(adj_csr_raw)
                    if uniform:
                        adj_csr.data = np.ones(len(adj_csr.data))

                    self.adjs.append(
                        sparse_mx_to_torch_sparse_tensor(
                            sp.csr_matrix(adj_csr / adj_csr.sum(axis=1))
                        ).to(device)
                    )
                if LOAD:
                    torch.save(
                        obj=self.adjs,
                        f=tmp,
                    )

        for i in range(self.n_hops + 1):
            self.ns.append(
                self.deepsets[i](
                    F.dropout(x, self.dropout, self.training),
                    adj=self.adjs[i] if i != 0 else None,
                )
            )
        return [self.readout(torch.stack(self.ns, dim=1), mean=True)]
        # return [torch.cat(self.ns, dim=1)]
        # return [self.mlp(F.dropout(torch.cat(self.ns, dim=1), self.dropout, self.training))]
