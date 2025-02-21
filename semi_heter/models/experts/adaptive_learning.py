"""Adaptive Learning"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
import copy
import math
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


class AdaptiveLearning(nn.Module):
    """Adaptive_learning"""

    def __init__(
        self,
        in_feats,
        h_feats,
        # NOTE: neighborhood_order >= 1 as ego node is the 1st order of the neighborhood
        n_gnn_layers: int,
        norm: str = "sym",
        renorm: bool = True,
        upth_st: float = 0.0015,
        upth_ed: float = 0.001,
        lowth_st: float = 0.1,
        lowth_ed: float = 0.5,
        upd: float = 10,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.upth_st = upth_st
        self.upth_ed = upth_ed
        self.lowth_st = lowth_st
        self.lowth_ed = lowth_ed
        self.upd = upd
        self.up_eta = None
        self.low_eta = None
        self.norm = norm
        self.renorm = renorm
        self.n_gnn_layers = n_gnn_layers

        self.pos_inds = None
        self.neg_inds = None
        self.upth = None
        self.lowth = None
        self.bs = None

        self.sm_fea_s = None
        self.dropout = dropout

        # attribute homophily: adaptive learning
        self.ec = MLP(in_feats=in_feats, h_feats=[h_feats], layers=1, acts=[nn.ReLU()])
        self.pair_dc = SampleDecoder(act=lambda x: x)

        self.initialized = False
        self.optimizer = None

    def loss(self, adj_preds, adj_labels):
        """compute loss

        Args:
            adj_preds (torch.Tensor):reconstructed adj

        Returns:
            torch.Tensor: loss
        """
        return F.binary_cross_entropy_with_logits(adj_preds, adj_labels)

    def forward(self, x):
        return self.ec(F.dropout(x, self.dropout, training=self.training))

    def init_batch(
        self,
        graph,
        n_epochs,
        bs,
        device=torch.device("cpu"),
    ):
        self.up_eta = (self.upth_ed - self.upth_st) / (n_epochs / self.upd)
        self.low_eta = (self.lowth_ed - self.lowth_st) / (n_epochs / self.upd)

        load_init = False
        file = Path(
            f"tmp/al/{graph.name}_{self.n_gnn_layers}_{self.upth_st}_{self.lowth_st}_{self.up_eta}_{self.low_eta}"
        )

        if load_init and file.exists():
            (
                bs,
                pos_inds,
                neg_inds,
                upth,
                lowth,
            ) = torch.load(file, map_location=device)
            self.generate_graph(
                pos_inds=pos_inds,
                n_nodes=graph.num_nodes(),
                x=graph.ndata["feat"],
                device=graph.device,
            )

            return (
                bs,
                pos_inds,
                neg_inds,
                upth,
                lowth,
            )

        make_parent_dirs(file)

        pos_inds, neg_inds = update_similarity(
            graph.ndata["feat"].data.cpu().numpy(),
            self.upth_st,
            self.lowth_st,
        )
        upth, lowth = update_threshold(
            self.upth_st,
            self.lowth_st,
            self.up_eta,
            self.low_eta,
        )
        self.generate_graph(
            pos_inds=pos_inds,
            n_nodes=graph.num_nodes(),
            x=graph.ndata["feat"],
            device=graph.device,
        )

        bs = min(bs, len(pos_inds))

        torch.save(
            obj=(
                bs,
                pos_inds,
                neg_inds,
                upth,
                lowth,
            ),
            f=file,
            pickle_protocol=4,
        )

        return (
            bs,
            pos_inds,
            neg_inds,
            upth,
            lowth,
        )

    def generate_graph(self, pos_inds, n_nodes, x, device):
        pos_xind = torch.LongTensor(pos_inds) // n_nodes
        pos_yind = torch.LongTensor(pos_inds) % n_nodes
        adj_nsl = (
            dgl.graph((pos_xind, pos_yind), num_nodes=n_nodes)
            .to_simple()
            .remove_self_loop()
            .adj_external(scipy_fmt="csr")
            .toarray()
        )
        # adj_nsl = (
        #     dgl.knn_graph(x.cpu(),k=10).to_simple().remove_self_loop().adj_external(
        #                   scipy_fmt="csr"
        #               ).toarray()
        # )
        adj_norm_s = preprocess_graph(
            adj_nsl=adj_nsl,
            layer=self.n_gnn_layers,
            norm="sym",
            renorm=True,
        )

        sm_fea_s = x.cpu()
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
        self.sm_fea_s = torch.FloatTensor(sm_fea_s).to(device)

    def update_batch(
        self,
        epoch,
        graph,
        pos_inds,
        neg_inds,
        bs,
        upth,
        lowth,
    ):
        if (epoch + 1) % self.upd == 0:
            with torch.no_grad():
                mu = self.forward(graph.ndata["feat"])
            hidden_emb = mu.cpu().data.numpy()
            upth, lowth = update_threshold(
                upth,
                lowth,
                self.up_eta,
                self.low_eta,
            )
            pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth)
            self.generate_graph(
                pos_inds=pos_inds,
                n_nodes=graph.num_nodes(),
                x=graph.ndata["feat"],
                device=graph.device,
            )
            bs = min(bs, len(pos_inds))
        return (
            pos_inds,
            neg_inds,
            bs,
            upth,
            lowth,
        )

    @staticmethod
    def sample_batch(
        st,
        ed,
        neg_inds,
        pos_inds,
        n_nodes,
        device=torch.device("cpu"),
    ):
        sampled_neg = torch.LongTensor(
            np.random.choice(
                neg_inds,
                size=(ed - st),
            )
        ).to(device)
        pos_inds = torch.LongTensor(pos_inds).to(device)
        sampled_inds = torch.cat((pos_inds[st:ed], sampled_neg), 0)
        xind = sampled_inds // n_nodes
        yind = sampled_inds % n_nodes
        return xind, yind

    def batch_forward(
        self,
        graph: dgl.DGLGraph,
        x_batch: torch.LongTensor,
        y_batch: torch.LongTensor,
        device=torch.device("cpu"),
        z=None,
    ):
        features = graph.ndata["feat"] if self.sm_fea_s is None else self.sm_fea_s
        batch_size = len(x_batch)
        zx = self.forward(torch.index_select(features, 0, x_batch)) if z is None else z[:batch_size]
        zy = self.forward(torch.index_select(features, 0, y_batch)) if z is None else z[batch_size:]

        batch_label = torch.cat(
            (
                torch.ones(batch_size // 2),
                torch.zeros(batch_size // 2),
            )
        ).to(device)
        batch_pred = self.pair_dc(
            zx,
            zy,
        )

        return self.loss(adj_preds=batch_pred, adj_labels=batch_label)

    def full_batch_forward(
        self,
        graph,
        bs,
        n_epochs,
        epoch,
        device: torch.device = torch.device("cpu"),
    ):
        if not self.initialized or epoch == 0:
            (
                self.bs,
                self.pos_inds,
                self.neg_inds,
                self.upth,
                self.lowth,
            ) = self.init_batch(
                graph=graph,
                n_epochs=n_epochs,
                bs=bs,
                device=device,
            )
        st = 0
        ed = self.bs

        length = len(self.pos_inds)
        loss = 0
        while ed <= length:
            xind, yind = self.sample_batch(
                st=st,
                ed=ed,
                neg_inds=self.neg_inds,
                pos_inds=self.pos_inds,
                n_nodes=graph.num_nodes(),
                device=device,
            )

            loss = loss + self.batch_forward(
                graph=graph,
                x_batch=xind,
                y_batch=yind,
                device=device,
            )

            st = ed
            if ed < length <= ed + self.bs:
                ed += length - ed
            else:
                ed += self.bs

        (
            self.pos_inds,
            self.neg_inds,
            self.bs,
            self.upth,
            self.lowth,
        ) = self.update_batch(
            epoch,
            graph,
            self.pos_inds,
            self.neg_inds,
            self.bs,
            self.upth,
            self.lowth,
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

                if not self.initialized or epoch == 0:
                    (
                        self.bs,
                        self.pos_inds,
                        self.neg_inds,
                        self.upth,
                        self.lowth,
                    ) = self.init_batch(
                        graph=graph,
                        n_epochs=n_epochs,
                        bs=batch_size,
                        device=device,
                    )
                st = 0
                ed = self.bs
                cur_loss = 0.0

                length = len(self.pos_inds)
                while ed <= length:
                    xind, yind = self.sample_batch(
                        st=st,
                        ed=ed,
                        neg_inds=self.neg_inds,
                        pos_inds=self.pos_inds,
                        n_nodes=graph.num_nodes(),
                        device=device,
                    )

                    loss = self.batch_forward(
                        graph=graph,
                        x_batch=xind,
                        y_batch=yind,
                        device=device,
                    )
                    loss.backward()
                    self.optimizer.step()
                    cur_loss += loss.item()

                    st = ed
                    if ed < length <= ed + self.bs:
                        ed += length - ed
                    else:
                        ed += self.bs

                (
                    self.pos_inds,
                    self.neg_inds,
                    self.bs,
                    self.upth,
                    self.lowth,
                ) = self.update_batch(
                    epoch,
                    graph,
                    self.pos_inds,
                    self.neg_inds,
                    self.bs,
                    self.upth,
                    self.lowth,
                )

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
                # self.warmup_edge_reconstruction(graph=graph, device=device)
                # embeddings = self.ec_er_sho(self.fs[0])
                # self.warmup_experts(graph=graph, device=device)

                torch.save(obj=self.state_dict(), f=state, pickle_protocol=4)


def update_similarity(z, upper_threshold, lower_treshold):
    """update similarity

    Args:
        z (numpy.ndarray):hidden embedding
        upper_threshold (float): upper threshold
        lower_treshold (float):lower treshold
        pos_num (int):number of positive samples
        neg_num (int):number of negative samples

    Returns:
        numpy.ndarray: list of positive indexs
        numpy.ndarray: list of negative indexs
    """
    cosine = np.matmul(z, np.transpose(z))
    cosine = cosine.reshape(
        [
            -1,
        ]
    )
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1 - lower_treshold) * len(cosine))

    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds)


def update_similarity_cos(cosine, upper_threshold, lower_treshold, pos_num, neg_num):
    """update similarity

    Args:
        z (numpy.ndarray):hidden embedding
        upper_threshold (float): upper threshold
        lower_treshold (float):lower treshold
        pos_num (int):number of positive samples
        neg_num (int):number of negative samples

    Returns:
        numpy.ndarray: list of positive indexs
        numpy.ndarray: list of negative indexs
    """
    # f_adj = np.matmul(z, np.transpose(z))
    # cosine = f_adj
    # cosine = cosine.reshape(
    #     [
    #         -1,
    #     ]
    # )
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1 - lower_treshold) * len(cosine))

    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds)


def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    """update threshold

    Args:
        upper_threshold (float): upper threshold
        lower_treshold (float):lower treshold
        up_eta (float):update step size of upper threshold
        low_eta (float):update step size of lower threshold

    Returns:
        upth (float): updated upth
        lowth (float): updated lowth
    """
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth
