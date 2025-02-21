"""Self Reconstruction"""

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
from ...utils import preprocess_graph


class SelfReconstruction(nn.Module):
    """Adaptive_learning"""

    def __init__(
        self,
        in_feats,
        h_feats,
        dropout=0.5,
    ) -> None:
        super().__init__()

        self.h_feats = h_feats
        # attribute heterophily: AE
        self.ec = MLP(in_feats=in_feats, h_feats=[h_feats], layers=1, acts=[nn.ReLU()])
        self.dc = LinTrans(1, [h_feats, in_feats])
        self.optimizer = None
        self.dropout = dropout

    def loss(self, d, features):
        return F.mse_loss(
            d,
            features,
            reduction="mean",
        )

    def forward(self, x):
        return self.ec(F.dropout(x, self.dropout, training=self.training))
        # return self.ec(x)

    def full_batch_forward(
        self,
        graph,
        z=None,
        device: torch.device = torch.device("cpu"),
    ):
        z = self.forward(graph.ndata["feat"]) if z is None else z
        d = self.dc(z)
        loss = self.loss(
            d=d,
            features=graph.ndata["feat"],
        )
        return loss

    def batch_forward(
        self,
        graph: dgl.DGLGraph,
        nodes_batch: np.array,
        device=torch.device("cpu"),
        z=None,
    ):
        batch_feat = torch.index_select(
            graph.ndata["feat"],
            0,
            torch.LongTensor(nodes_batch).to(device),
        )
        if z is None:
            z = self.forward(batch_feat)
        d = self.dc(z)
        return self.loss(
            d=d,
            features=batch_feat,
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
