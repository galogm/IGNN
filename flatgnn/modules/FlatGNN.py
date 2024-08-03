"""FaltGNN Layer."""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
import copy
import math
import random
import time
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score as ACC
from sklearn.preprocessing import normalize
from the_utils import get_str_time
from the_utils import make_parent_dirs
from the_utils import save_to_csv_files
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ModuleList
from torch.utils.tensorboard import SummaryWriter
from torch_sparse import fill_diag
from torch_sparse import SparseTensor
from tqdm import tqdm

from ..utils import preprocess_neighborhoods
from .DeepSets import DeepSets
from .LSTM import LSTM
from .MLP import MLP


class FlatGNN(nn.Module):
    """FlatGNN."""
    def __init__(
        self,
        in_feats,
        h_feats,
        n_hops,
        dropout=0.0,
        n_intervals=3,
        no_save=False,
        nie="deepsets",
        nrl="concat",
    ):
        super().__init__()
        self.no_save = no_save
        self.n_hops = n_hops
        self.h_feats = h_feats
        self.nie = nie
        self.nrl = nrl

        N_NIE = self.n_hops + 1
        if nie == "deepsets":
            self.nei_ind_emb = nn.ModuleList(
                [
                    DeepSets(
                        in_feats=in_feats,
                        h_feats=h_feats,
                        dropout=dropout,
                    ) for _ in range(N_NIE)
                ]
            )
        elif nie == "gcn":
            self.nei_ind_emb = nn.ModuleList(
                [
                    MLP(
                        in_feats=in_feats,
                        h_feats=[h_feats],
                        acts=[nn.ReLU()],
                        dropout=dropout,
                    ) for _ in range(N_NIE)
                ]
            )

        self.nei_feats = None

        self.interval = n_intervals
        self.n_relations = self.n_hops + 1 - self.interval

        if nrl == "concat":
            self.nei_rel_learn = nn.ModuleList(
                [
                    MLP(
                        in_feats=h_feats * N_NIE,
                        h_feats=[h_feats],
                        layers=1,
                        acts=[nn.ReLU()],
                        dropout=dropout,
                    )
                ],
            )
        elif nrl == "multi-con":
            self.nei_rel_learn = nn.ModuleList(
                [
                    MLP(
                        in_feats=h_feats * self.interval,
                        h_feats=[h_feats],
                        layers=1,
                        acts=[nn.ReLU()],
                        dropout=dropout,
                    ) for _ in range(self.n_relations + 1)
                ]
            )
        elif nrl in ["max", "sum", "mean"]:
            self.nei_rel_learn = nn.ModuleList(
                [
                    MLP(
                        in_feats=h_feats,
                        h_feats=[h_feats],
                        layers=1,
                        acts=[nn.ReLU()],
                        dropout=dropout,
                    )
                ],
            )
        elif nrl == "lstm":
            self.nei_rel_learn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(dropout),
                        LSTM(
                            input_dim=h_feats,
                            hidden_dim=h_feats,
                            output_dim=h_feats,
                            num_layers=1,
                            dropout=dropout,
                        ),
                    )
                ],
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
                    ) if self.adj is None else self.adj
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
            self.ns.append(self.nei_ind_emb[i](self.nei_feats[i]))

        hops_feats = torch.cat(self.ns, dim=1)

        if self.nrl == "none":
            return [hops_feats]

        if self.nrl == "concat":
            return [
                torch.cat(
                    [nei_rel_learn(hops_feats) for nei_rel_learn in self.nei_rel_learn],
                    dim=1,
                )
            ]
        if self.nrl == "multi-con":
            return [
                torch.cat(
                    [
                        self.nei_rel_learn[i](
                            hops_feats[:, self.h_feats * i:self.h_feats * (i + self.interval)]
                        ) for i in range(self.n_relations + 1)
                    ],
                    dim=1,
                )
            ]
        if self.nrl == "max":
            return [self.nei_rel_learn[0](torch.max(
                torch.stack(self.ns, dim=1),
                dim=1,
            )[0])]
        if self.nrl == "mean":
            return [self.nei_rel_learn[0](torch.mean(
                torch.stack(self.ns, dim=1),
                dim=1,
            ))]
        if self.nrl == "sum":
            return [self.nei_rel_learn[0](torch.sum(
                torch.stack(self.ns, dim=1),
                dim=1,
            ))]
        if self.nrl == "lstm":
            return [self.nei_rel_learn[0](torch.stack(self.ns, dim=1), )]
