"""Neighborhood Prediction"""

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
from the_utils import make_parent_dirs
from torch import nn
from torch.distributions.normal import Normal

from semi_heter.utils.common import (
    sparse_mx_to_torch_sparse_tensor,
    sys_normalized_adjacency,
)

from ...baselines import H2GCN
from ...modules import MLP, Data, InnerProductDecoder, LinTrans, SampleDecoder
from ...utils import preprocess_graph


class NeighborhoodPrediction(nn.Module):
    """Adaptive_learning"""

    def __init__(
        self,
        in_feats,
        h_feats,
        # NOTE: neighborhood_order >= 1 as ego node is the 1st order of the neighborhood
        neighborhood_order,
        n_clusters,
        device,
        dropout=0.5,
        alpha=0.5,
        k=2,
    ) -> None:
        super().__init__()
        self.h_feats = h_feats
        self.alpha = alpha
        self.neighborhood_order = neighborhood_order
        # structure heter: neighborhood prediction
        # self.ec = MLP(in_feats=in_feats, h_feats=[h_feats], layers=1, acts=[nn.ReLU()])
        self.ec = H2GCN(
            in_features=in_feats,
            class_num=n_clusters,
            device=device,
            is_layer=True,
            args=Data(
                **{
                    "use_relu": True,
                    "hidden_dim": h_feats,
                    "k": k,
                    "dropout": dropout,
                }
            ),
        )

        self.dc = LinTrans(1, [h_feats, in_feats])

        self.initialized = False
        self.optimizer = None

    def split_frequencies(self, adj_nsl):
        # TODO: try without symmetric normalization
        I = torch.eye(adj_nsl.shape[0]).to(self.device)
        adj_ = copy.deepcopy(adj_nsl + I).to(self.device)
        _D = torch.diag(adj_.sum(1) * (-0.5))
        tilde_A = _D.matmul(adj_).matmul(_D)

        adj_d = I - tilde_A
        adj_m = tilde_A

        return adj_m.to(self.device), adj_d.to(self.device)

    def flatten_neighborhood(
        self,
        adj_nsl: torch.Tensor,
        features: torch.Tensor,
        save_dir: str = None,
        dataset: str = None,
    ) -> torch.Tensor:
        """Flatten the neighborhoods

        Args:
            adj_nsl (torch.Tensor): adj without self-loops.
            features (torch.Tensor): features.
            k (int): the order of neighborhood to be flattened.

        Returns:
            torch.Tensor: the list of features of k-th order neighborhood.
        """
        k = self.neighborhood_order
        save_path = (
            Path(save_dir).joinpath(dataset).joinpath(f"{k}") if save_dir and dataset else None
        )

        if save_path and save_path.exists():
            with open(save_path, "rb") as f:
                fsm, fsd = torch.load(f=f, map_location=self.device)
            return fsm, fsd

        fsm = [features]
        fsd = [features]
        adj_m, adj_d = self.split_frequencies(adj_nsl)

        for _ in range(k - 1) if k >= 2 else range(k):
            fsm.append(F.normalize(adj_m @ features, dim=1).to(self.device))
            fsd.append(F.normalize(adj_d @ features, dim=1).to(self.device))

            adj_m, adj_d = self.split_frequencies(adj_m)

        fsm, fsd = torch.stack(fsm, dim=0), torch.stack(fsd, dim=0)

        if save_path:
            make_parent_dirs(save_path)
            with open(save_path, "wb") as f:
                torch.save(
                    [fsm, fsd],
                    f=f,
                    pickle_protocol=4,
                )

        return fsm, fsd

    def get_neighborhood_features(
        self,
        graph,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        adj_nsl = graph.remove_self_loop().adj_external(scipy_fmt="csr").toarray()
        features = graph.ndata["feat"].to(device)
        self.fsm, self.fsd = self.flatten_neighborhood(
            adj_nsl=torch.Tensor(adj_nsl).to(device),
            features=features,
            save_dir="./tmp",
            dataset=graph.name,
        )
        self.adj_tensor = sparse_mx_to_torch_sparse_tensor(
            sys_normalized_adjacency(
                graph.adj_external(scipy_fmt="csr") + sp.eye(graph.num_nodes())
            )
        ).to(self.device)

    def forward(self, features):
        z = self.ec(features, self.adj_tensor, return_Z=True)
        return z

    def loss(self, d, hp, lp):
        return (1 - self.alpha) * F.mse_loss(
            d,
            hp,
            reduction="mean",
        ) + self.alpha * F.mse_loss(
            d,
            lp,
            reduction="mean",
        )

    def init_batch(
        self,
        graph,
        device: torch.device = torch.device("cpu"),
    ):
        self.get_neighborhood_features(
            graph=graph,
            device=device,
        )

        self.initialized = True

    def batch_forward(
        self,
        graph: dgl.DGLGraph,
        nodes_batch: np.array,
        device=torch.device("cpu"),
        z=None,
    ):
        if not self.initialized:
            self.init_batch(graph=graph, device=device)

        batch_idx = torch.LongTensor(nodes_batch).to(device)
        if z is None:
            z = torch.index_select(
                self.forward(graph.ndata["feat"]),
                0,
                batch_idx,
            )

        d = self.dc(z)

        return self.loss(
            d,
            hp=torch.index_select(self.fsd[1].to(device), 0, batch_idx),
            lp=torch.index_select(self.fsm[1].to(device), 0, batch_idx),
        )

    def full_batch_forward(
        self,
        graph,
        z=None,
        device: torch.device = torch.device("cpu"),
    ):
        z = self.forward(graph.ndata["feat"].to(device)) if z is None else z
        d = self.dc(z)

        loss = self.loss(
            d,
            hp=self.fsd[1],
            lp=self.fsm[1],
        )
        return loss

    def fit(
        self,
        graph,
        lr=0.001,
        n_epochs=100,
        batch_size=2048,
        load_state=False,
        state=None,
        device: torch.device = torch.device("cpu"),
    ):
        if load_state and Path(state).exists():
            print(f"load {state}")
            self.load_state_dict(torch.load(state, map_location=device))
        else:
            make_parent_dirs(Path(state))

            best_loss = 1e9
            cnt = 0
            best_epoch = 0
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            self.to(device)
            for epoch in range(n_epochs):
                self.train()
                t = time.time()
                self.optimizer.zero_grad()

                N_NODES = graph.num_nodes()
                nodes = np.array(range(N_NODES))
                random.shuffle(nodes)

                st = 0
                ed = min(batch_size, N_NODES)
                cur_loss = 0.0
                while ed <= N_NODES:
                    loss = self.batch_forward(
                        graph=graph,
                        nodes_batch=nodes[st:ed],
                        device=device,
                    )
                    loss.backward()
                    self.optimizer.step()
                    cur_loss += loss.item()
                    st = ed
                    if ed < N_NODES <= ed + batch_size:
                        ed += N_NODES - ed
                    else:
                        ed += batch_size

                if epoch % 10 == 0:
                    print(f"Epoch: {epoch}, loss={cur_loss:.5f}, time={time.time() - t:.5f}")

                if cur_loss < best_loss:
                    cnt = 0
                    best_epoch = epoch
                    best_loss = cur_loss
                    # if self.best_model:
                    #     self.best_model = None
                    # self.best_model = copy.deepcopy(self).to(self.device)

                else:
                    cnt += 1
                    # print(f"loss increase count:{cnt}")
                    if cnt >= 200:
                        print(f"early stopping,best epoch:{best_epoch}")
                        break

            torch.save(obj=self.state_dict(), f=state, pickle_protocol=4)
