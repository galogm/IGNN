"""Edge Reconstruction"""

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

from ...modules import MLP, InnerProductDecoder, LinTrans, SampleDecoder
from ...utils import get_batch_edges, preprocess_graph


class EdgeReconstruction(nn.Module):
    """Homophilous GNN"""

    def __init__(
        self,
        in_feats,
        h_feats,
        n_gnn_layers=0,
        dropout=0.5,
    ) -> None:
        super().__init__()

        self.pos_weight = None
        self.norm_weights = None
        self.lbls = None
        self.adj_norm_s = None
        self.sm_fea_s: torch.tensor = None
        self.n_gnn_layers = n_gnn_layers
        self.h_feats = h_feats
        self.dropout = dropout

        self.ec = MLP(in_feats=in_feats, h_feats=[h_feats], layers=1, acts=[nn.ReLU()])

        self.dc = InnerProductDecoder(act=lambda x: x)
        self.pair_dc = SampleDecoder(act=lambda x: x)
        self.optimizer = None

        self.initialized = False

    def forward(self, x):
        return self.ec(F.dropout(x, self.dropout, training=self.training))

    @staticmethod
    def loss(preds, labels, norm=1.0, pos_weight=None):
        return norm * F.binary_cross_entropy_with_logits(
            preds,
            labels,
            pos_weight=pos_weight,
        )

    @staticmethod
    def loss_pair(adj_preds, adj_labels):
        """compute loss

        Args:
            adj_preds (torch.Tensor):reconstructed adj

        Returns:
            torch.Tensor: loss
        """
        return F.binary_cross_entropy_with_logits(adj_preds, adj_labels)

    def init_batch(
        self,
        graph,
        norm: str = "sym",
        renorm: bool = True,
        device=torch.device("cpu"),
    ):
        adj_nsl = graph.remove_self_loop().adj_external(scipy_fmt="csr")
        self.adj_norm_s = preprocess_graph(
            adj_nsl=adj_nsl,
            layer=self.n_gnn_layers,
            norm=norm,
            renorm=renorm,
        )

        sm_fea_s = graph.ndata["feat"].cpu()
        for a in self.adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
        self.sm_fea_s = torch.FloatTensor(sm_fea_s).to(device)

        n_nodes = adj_nsl.shape[0]
        n_edges = adj_nsl.sum()
        self.pos_weight = torch.FloatTensor([(float(n_nodes * n_nodes - n_edges) / n_edges)]).to(
            device
        )
        self.norm_weights = n_nodes * n_nodes / float((n_nodes * n_nodes - n_edges) * 2)
        self.lbls = torch.FloatTensor(adj_nsl.toarray()).view(-1).to(device)

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

        (
            ((pos_u, pos_v), (pos_rows, pos_cols)),
            ((neg_u, neg_v), (neg_rows, neg_cols)),
        ) = get_batch_edges(
            adj_csr=graph.add_self_loop().adj_external(scipy_fmt="csr"),
            nodes_batch=nodes_batch,
            sample_neg=True,
        )

        adj_labels = torch.cat(
            (
                torch.ones(len(pos_u)),
                torch.zeros(len(neg_u)),
            )
        ).to(device)
        if z is None:
            return self.loss_pair(
                adj_preds=self.pair_dc(
                    self.forward(
                        torch.index_select(
                            self.sm_fea_s,
                            0,
                            torch.LongTensor(np.concatenate((pos_u, neg_u))).to(device),
                        )
                    ),
                    self.forward(
                        torch.index_select(
                            self.sm_fea_s,
                            0,
                            torch.LongTensor(np.concatenate((pos_v, neg_v))).to(device),
                        )
                    ),
                ),
                adj_labels=adj_labels,
            )
        return self.loss_pair(
            adj_preds=self.pair_dc(
                torch.index_select(
                    z,
                    0,
                    torch.LongTensor(np.concatenate((pos_rows, neg_rows))).to(device),
                ),
                torch.index_select(
                    z,
                    0,
                    torch.LongTensor(np.concatenate((pos_cols, neg_cols))).to(device),
                ),
            ),
            adj_labels=adj_labels,
        )

    def full_batch_forward(
        self,
        graph,
        z=None,
        device: torch.device = torch.device("cpu"),
    ):
        if not self.initialized:
            self.init_batch(graph=graph, device=device)
        z = self.forward(self.sm_fea_s) if z is None else z
        d = self.dc(z)
        return self.loss(
            preds=d.view(-1),
            labels=self.lbls,
            norm=self.norm_weights,
            pos_weight=self.pos_weight,
        )

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
                        nodes_batch=torch.LongTensor(nodes[st:ed]),
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
