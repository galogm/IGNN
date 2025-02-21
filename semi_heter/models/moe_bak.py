"""MoE"""

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

from ..modules import MLP, InnerProductDecoder, LinTrans, SampleDecoder
from ..utils import preprocess_graph


def sk_clustering(
    X: torch.Tensor,
    n_clusters: int,
    name: str = "kmeans",
) -> np.ndarray:
    """sklearn clustering.

    Args:
        X (torch.Tensor): data embeddings.
        n_clusters (int): num of clusters.
        name (str, optional): type name. Defaults to 'kmeans'.

    Raises:
        NotImplementedError: clustering method not implemented.

    Returns:
        np.ndarray: cluster assignments.
    """
    if name == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init="auto")
        label_pred = model.fit(X).labels_
        return label_pred

    # if name == "spectral":
    #     model = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
    #     label_pred = model.fit(X).labels_
    #     return label_pred

    raise NotImplementedError


class MoE(nn.Module):
    """MoE"""

    def __init__(
        self,
        in_feats,
        h_feats,
        # NOTE: neighborhood_order >= 1 as ego node is the 1st order of the neighborhood
        neighborhood_order,
        n_gnn_layers: int,
        n_clusters,
        n_epochs: int = 400,
        lr: float = 0.001,
        early_stop: int = 100,
        device=None,
        act=nn.ReLU,
        normalize=True,
        alpha=0.5,
        high_ratio=1,
        low_ratio=1,
        n_experts=4,
        noisy_gating=True,
        norm: str = "sym",
        renorm: bool = True,
        upth_st: float = 0.0015,
        upth_ed: float = 0.001,
        lowth_st: float = 0.1,
        lowth_ed: float = 0.5,
        upd: float = 10,
        bs: int = 10000,
    ) -> None:
        super().__init__()

        self.upth_st = upth_st
        self.upth_ed = upth_ed
        self.lowth_st = lowth_st
        self.lowth_ed = lowth_ed
        self.upd = upd
        self.bs = bs
        self.norm = norm
        self.renorm = renorm

        # MoE
        self.noisy_gating = noisy_gating
        self.k = self.n_experts = n_experts
        self.loss_coef = alpha
        # instantiate experts
        self.w_gate = nn.Parameter(
            torch.zeros(h_feats * self.n_experts, self.n_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(h_feats * self.n_experts, self.n_experts), requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        # assert(self.k <= self.n_experts)

        # URL
        self.lr = lr
        self.n_epochs = n_epochs
        self.estop_steps = early_stop
        self.n_clusters = n_clusters
        self.device = device
        self.normalize = normalize
        self.alpha = alpha
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio
        self.best_model = None
        self.h_feats = h_feats

        self.neighborhood_order = neighborhood_order
        self.n_gnn_layers = n_gnn_layers

        # attribute homophily: adaptive learning
        self.ec_al_aho = MLP(in_feats=in_feats, h_feats=[h_feats], layers=1, acts=[nn.ReLU()])

        # attribute heterophily: AE
        self.ec_sr_ahe = MLP(in_feats=in_feats, h_feats=[h_feats], layers=1, acts=[nn.ReLU()])

        # structure hom: SGC
        self.ec_er_sho = MLP(in_feats=in_feats, h_feats=[h_feats], layers=1, acts=[nn.ReLU()])

        # structure heter: neighborhood prediction
        self.ec_np_she = MLP(in_feats=in_feats, h_feats=[h_feats], layers=1, acts=[nn.ReLU()])

        # representation hom: adaptive learning
        self.ec_al_rho = MLP(
            in_feats=h_feats * self.n_experts, h_feats=[h_feats], layers=1, acts=[nn.ReLU()]
        )

        self.dc_np_she = LinTrans(1, [h_feats, in_feats])
        self.dc_sr_ahe = LinTrans(1, [h_feats, in_feats])

        self.classifier = MLP(
            in_feats=h_feats, h_feats=[n_clusters], layers=1, acts=[nn.Softmax(dim=1)]
        )

        self.ce = torch.nn.CrossEntropyLoss()

        self.inner_product_decoder = InnerProductDecoder(act=lambda x: x)
        self.pair_decoder = SampleDecoder(act=lambda x: x)

        self.warmup_prepared = False
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def split_frequencies(self, adj):
        # TODO: try without symmetric normalization
        I = torch.eye(adj.shape[0]).to(self.device)
        adj_ = copy.deepcopy(adj + I).to(self.device)
        _D = torch.diag(adj_.sum(1) * (-0.5))
        tilde_A = _D.matmul(adj_).matmul(_D)

        adj_d = I - tilde_A
        adj_m = tilde_A

        return adj_m.to(self.device), adj_d.to(self.device)

    def flatten_neighborhood(
        self,
        adj: torch.Tensor,
        features: torch.Tensor,
        save_dir: str = None,
        dataset: str = None,
    ) -> torch.Tensor:
        """Flatten the neighborhoods

        Args:
            adj (torch.Tensor): adj without self-loops.
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
        adj_m, adj_d = self.split_frequencies(adj)

        for _ in range(k - 1) if k >= 2 else range(k):
            fsm.append(F.normalize(adj_m @ features, dim=1).to(self.device))
            fsd.append(F.normalize(adj_d @ features, dim=1).to(self.device))

            adj_m, adj_d = self.split_frequencies(adj_m)

        fsm, fsd = torch.stack(fsm, dim=0), torch.stack(fsd, dim=0)

        if save_path:
            make_parent_dirs(save_path)
            with open(save_path, "wb") as f:
                torch.save([fsm, fsd], f=f, pickle_protocol=4)

        return fsm, fsd

    def forward_al_aho(self, x, y):
        zx = self.ec_al_aho(x)
        zy = self.ec_al_aho(y)

        return self.pair_decoder(zx, zy)

    def forward_sr_ahe(self, features):
        z = self.ec_sr_ahe(features)
        d = self.dc_sr_ahe(z)
        return z, d

    def forward_er_sho(self, features):
        z = self.ec_er_sho(features)
        d = self.inner_product_decoder(z)
        return z, d

    def forward_np_she(self, features):
        z = self.ec_np_she(features)
        d = self.dc_np_she(z)
        return z, d

    def forward_al_rho(self, x, y):
        zx1 = self.ec_al_aho(x)
        zy1 = self.ec_al_aho(y)

        zx2 = self.ec_sr_ahe(x)
        zy2 = self.ec_sr_ahe(y)

        zx3 = self.ec_er_sho(x)
        zy3 = self.ec_er_sho(y)

        zx4 = self.ec_np_she(x)
        zy4 = self.ec_np_she(y)

        gates = self.noisy_top_k_gating(x, train=True)
        return (
            gates[:, 0] * self.pair_decoder(zx1, zy1)
            + gates[:, 1] * self.pair_decoder(zx2, zy2)
            + gates[:, 2] * self.pair_decoder(zx3, zy3)
            + gates[:, 3] * self.pair_decoder(zx4, zy4)
        )

    def forward_kernel(self, x, coef=0.25):
        zx1 = self.ec_al_aho(x)

        zx2 = self.ec_sr_ahe(x)

        zx3 = self.ec_er_sho(x)

        zx4 = self.ec_np_she(x)

        return (
            coef * zx1.matmul(zx1.T)
            + coef * zx2.matmul(zx2.T)
            + coef * zx3.matmul(zx3.T)
            + coef * zx4.matmul(zx4.T)
        ).view(-1)

    def forward(self, x, coef=None, frozen_experts=False):
        if frozen_experts:
            with torch.no_grad():
                zx1 = self.ec_al_aho(x)

                zx2 = self.ec_sr_ahe(x)

                zx3 = self.ec_er_sho(x)

                zx4 = self.ec_np_she(x)
        else:
            zx1 = self.ec_al_aho(x)

            zx2 = self.ec_sr_ahe(x)

            zx3 = self.ec_er_sho(x)

            zx4 = self.ec_np_she(x)

        if coef is None:
            # self.gates = self.noisy_top_k_gating(x, train=True)
            self.gates = self.noisy_top_k_gating(
                torch.concat((zx1, zx2, zx3, zx4), dim=1), train=True
            )
            return (
                self.gates[:, 0].unsqueeze(dim=1) * zx1
                + self.gates[:, 1].unsqueeze(dim=1) * zx2
                + self.gates[:, 2].unsqueeze(dim=1) * zx3
                + self.gates[:, 3].unsqueeze(dim=1) * zx4
            )
        return coef * zx1 + coef * zx2 + coef * zx3 + coef * zx4

    @staticmethod
    def loss_er(preds, labels, norm=1.0, pos_weight=None):
        return norm * F.binary_cross_entropy_with_logits(
            preds,
            labels,
            pos_weight=pos_weight,
        )

    def loss_np(self, d, hp, lp):
        return (1 - self.alpha) * F.mse_loss(
            d[self.idx_h],
            hp[self.idx_h],
            reduction="mean",
        ) + self.alpha * F.mse_loss(
            d[self.idx_l],
            lp[self.idx_l],
            reduction="mean",
        )

    def loss_sr(self, d, features):
        return F.mse_loss(
            d,
            features,
            reduction="mean",
        )

    def get_neighborhood_features(self, graph, adj=None):
        adj = np.array(adj) if adj is not None else graph.adj_external(scipy_fmt="csr").toarray()
        self.features = graph.ndata["feat"].to(self.device)

        self.adj_nsl = torch.Tensor(adj).to(self.device)

        self.fsm, self.fsd = self.flatten_neighborhood(
            adj=self.adj_nsl,
            features=self.features,
            save_dir="./tmp",
            dataset=graph.name,
        )
        self.adj_nsl = self.adj_nsl + torch.eye(self.adj_nsl.shape[0]).to(self.device)
        self.fs = F.normalize(self.fsm, dim=2) if self.normalize else self.fsm
        self.features = self.fs[0]

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, n_experts]
          load: a Tensor with shape [n_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.n_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        return gates

        # if self.noisy_gating and self.k < self.n_experts and train:
        #     load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(
        #         0
        #     )
        # else:
        #     load = self._gates_to_load(gates)
        # return gates, load

    def warmup_adaptive_learning(self, graph, adj=None, device=torch.device("cpu")):

        if not self.warmup_prepared:
            self.device = device
            self.n_edges = graph.num_edges()
            self.n_nodes = graph.num_nodes()
            self.get_neighborhood_features(graph, adj=adj)
            self.to(self.device)

        adj_norm_s = preprocess_graph(
            graph.adj_external(scipy_fmt="csr"),
            self.n_gnn_layers,
            norm=self.norm,
            renorm=self.renorm,
        )
        sm_fea_s = self.features.cpu().numpy()

        print("Laplacian Smoothing...")
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
        self.sm_fea_s = torch.FloatTensor(sm_fea_s).to(self.device)

        pos_num = self.n_edges
        neg_num = self.n_nodes * self.n_nodes - pos_num

        up_eta = (self.upth_ed - self.upth_st) / (self.n_epochs / self.upd)
        low_eta = (self.lowth_ed - self.lowth_st) / (self.n_epochs / self.upd)

        pos_inds, neg_inds = update_similarity(
            normalize(self.sm_fea_s.data.cpu().numpy()),
            self.upth_st,
            self.lowth_st,
            pos_num,
            neg_num,
        )
        upth, lowth = update_threshold(self.upth_st, self.lowth_st, up_eta, low_eta)

        bs = min(self.bs, len(pos_inds))
        # length = len(pos_inds)
        pos_inds_cuda = torch.LongTensor(pos_inds).to(self.device)

        best_loss = 1e9
        cnt = 0
        best_epoch = 0

        print("Start Training...")
        for epoch in range(self.n_epochs):
            st, ed = 0, bs
            batch_num = 0
            self.train()
            length = len(pos_inds)

            self.optimizer.zero_grad()
            loss = torch.tensor(0.0)
            t = time.time()
            while ed <= length:
                print(f"batch_num:{batch_num}")
                sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed - st)).to(
                    self.device
                )
                sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0)
                xind = sampled_inds // self.n_nodes
                yind = sampled_inds % self.n_nodes
                x = torch.index_select(self.sm_fea_s, 0, xind)
                y = torch.index_select(self.sm_fea_s, 0, yind)

                batch_label = torch.cat((torch.ones(ed - st), torch.zeros(ed - st))).to(self.device)
                batch_pred = self.forward_al_aho(x, y)
                loss = loss + loss_function(adj_preds=batch_pred, adj_labels=batch_label)

                st = ed
                batch_num += 1
                if ed < length <= ed + bs:
                    ed += length - ed
                else:
                    ed += bs

            loss.backward()
            cur_loss = loss.item()
            self.optimizer.step()

            if (epoch + 1) % self.upd == 0:
                mu = self.ec_al_aho(self.sm_fea_s)
                hidden_emb = mu.cpu().data.numpy()
                upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
                pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num)
                bs = min(self.bs, len(pos_inds))
                pos_inds_cuda = torch.LongTensor(pos_inds).to(self.device)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, train_gae={cur_loss:.5f}, time={time.time() - t:.5f}")

            if cur_loss < best_loss:
                cnt = 0
                best_epoch = epoch
                best_loss = cur_loss
                # del self.best_model
                # self.best_model = copy.deepcopy(self.to(self.device))
                # self.embedding = mu.data.cpu().numpy()
                # self.memberships = kmeans.labels_
            else:
                cnt += 1
                print(f"loss increase count:{cnt}")
                if cnt >= self.estop_steps:
                    print(f"early stopping,best epoch:{best_epoch}")
                    break

    def warmup_self_reconstruction(self, graph, adj=None, device=torch.device("cpu")):

        if not self.warmup_prepared:
            self.device = device
            self.n_edges = graph.num_edges()
            self.n_nodes = graph.num_nodes()
            self.get_neighborhood_features(graph, adj=adj)
            self.to(self.device)

        best_loss = 1e9
        cnt = 0
        for epoch in range(self.n_epochs):
            self.train()
            self.optimizer.zero_grad()

            z, d = self.forward_sr_ahe(features=self.fs[0])
            loss = self.loss_sr(
                d=d,
                features=self.fs[0],
            )
            loss.backward()
            self.optimizer.step()

            cur_loss = loss.item()

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, embeds_loss={cur_loss}")

            if cur_loss < best_loss:
                cnt = 0
                best_epoch = epoch
                best_loss = cur_loss
                # if self.best_model:
                #     self.best_model = None
                # self.best_model = copy.deepcopy(self).to(self.device)
            else:
                cnt += 1
                if cnt >= self.estop_steps:
                    break

    def warmup_edge_reconstruction(self, graph, adj=None, device=torch.device("cpu")):

        if not self.warmup_prepared:
            self.device = device
            self.n_edges = graph.num_edges()
            self.n_nodes = graph.num_nodes()
            self.get_neighborhood_features(graph, adj=adj)
            self.to(self.device)

        self.pos_weight = torch.FloatTensor(
            [
                (
                    float(self.adj_nsl.shape[0] * self.adj_nsl.shape[0] - self.adj_nsl.sum())
                    / self.adj_nsl.sum()
                )
            ]
        ).to(self.device)
        self.norm_weights = (
            self.adj_nsl.shape[0]
            * self.adj_nsl.shape[0]
            / float((self.adj_nsl.shape[0] * self.adj_nsl.shape[0] - self.adj_nsl.sum()) * 2)
        )
        self.lbls = self.adj_nsl.view(-1).to(self.device)

        best_loss = 1e9
        cnt = 0
        for epoch in range(self.n_epochs):
            self.train()
            self.optimizer.zero_grad()

            z, d = self.forward_er_sho(features=self.fs[0])
            loss = self.loss_er(
                d.view(-1),
                self.lbls,
                norm=self.norm_weights,
                pos_weight=self.pos_weight,
            )

            loss.backward()

            self.optimizer.step()
            cur_loss = loss.item()

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, embeds_loss={cur_loss}")

            if cur_loss < best_loss:
                cnt = 0
                best_epoch = epoch
                best_loss = cur_loss
                # if self.best_model:
                #     self.best_model = None
                # self.best_model = copy.deepcopy(self).to(self.device)

            else:
                cnt += 1
                if cnt >= self.estop_steps:
                    break

    def warmup_neighbor_predict(
        self, graph, adj=None, high_ratio=None, low_ratio=None, device=torch.device("cpu")
    ):
        if not self.warmup_prepared:
            self.device = device
            self.n_edges = graph.num_edges()
            self.n_nodes = graph.num_nodes()
            self.get_neighborhood_features(graph, adj=adj)
            self.to(self.device)

        self.high_ratio = high_ratio or self.high_ratio
        self.low_ratio = low_ratio or self.low_ratio

        dd = self.adj_nsl.sum(1)
        self.idx_h = torch.argsort(dd, descending=True)[
            : int(self.features.shape[0] * self.high_ratio)
        ]
        self.idx_l = torch.argsort(dd, descending=True)[
            : int(self.features.shape[0] * self.low_ratio)
        ]

        self.to(self.device)
        best_loss = 1e9
        cnt = 0
        for epoch in range(self.n_epochs):
            self.train()
            self.optimizer.zero_grad()

            z, d = self.forward_np_she(features=self.fs[0])
            loss = self.loss_np(
                d,
                hp=self.fsd[1],
                lp=self.fsm[1],
            )

            loss.backward()

            self.optimizer.step()
            cur_loss = loss.item()

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, embeds_loss={cur_loss}")

            if cur_loss < best_loss:
                cnt = 0
                best_epoch = epoch
                best_loss = cur_loss
                # if self.best_model:
                #     self.best_model = None
                # self.best_model = copy.deepcopy(self).to(self.device)

            else:
                cnt += 1
                if cnt >= self.estop_steps:
                    break

    def warmup(self, graph, adj=None, device=torch.device("cpu")):
        self.warmup_adaptive_learning(graph, device=device)
        self.warmup_self_reconstruction(graph, device=device)
        self.warmup_edge_reconstruction(graph, device=device)
        self.warmup_neighbor_predict(graph, device=device)

    def pretrain_experts(
        self,
        graph,
        device: torch.device = torch.device("cpu"),
    ):
        if not self.warmup_prepared:
            self.device = device
            self.n_edges = graph.num_edges()
            self.n_nodes = graph.num_nodes()
            self.get_neighborhood_features(graph)
            self.to(self.device)
            self.warmup_prepared = True
            self.warmup(graph, device=device)

    def test(self, X, labels, index_list):
        with torch.no_grad():
            embeddings = self.forward(X, frozen_experts=True)
            logits_onehot = self.classifier(embeddings)
            y_pred = torch.argmax(logits_onehot, dim=1)
        acc_list = []
        for index in index_list:
            # print(index,labels[index].cpu(), y_pred[index].cpu())
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list

    def pretrain_gating(
        self,
        graph,
        labels,
        train_mask,
        val_mask,
        test_mask,
        device: torch.device = torch.device("cpu"),
    ):
        # with torch.no_grad():
        # embeddings_al = self.ec_al_aho(self.features)
        # embeddings_sr = self.ec_sr_ahe(self.features)
        # embeddings_er = self.ec_er_sho(self.features)
        # embeddings_np = self.ec_np_she(self.features)
        self.device = device
        self.n_edges = graph.num_edges()
        self.n_nodes = graph.num_nodes()
        self.get_neighborhood_features(graph)
        self.to(self.device)
        labels = labels.to(device)
        best_epoch = 0
        best_acc = 0.0
        cnt = 0
        best_state_dict = None

        t_start = time.time()
        for epoch in range(1000):
            self.train()
            self.optimizer.zero_grad()

            embeddings = self.forward(self.fs[0], frozen_experts=True)
            logits_onehot = self.classifier(embeddings)
            loss = self.ce(logits_onehot[train_mask], labels[train_mask])

            loss.backward()
            self.optimizer.step()

            [
                train_acc,
                valid_acc,
                test_acc,
            ] = self.test(
                self.features,
                labels,
                [train_mask, val_mask, test_mask],
            )
            print(
                f"epoch:{epoch},loss: {loss.item()}, train acc: {train_acc:.3f}, valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}"
            )
            if valid_acc > best_acc:
                cnt = 0
                best_acc = valid_acc
                best_acc_t = test_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
                # print(f"\nEpoch:{epoch}, Loss:{loss.item()}")
                # print(
                #     f"train acc: {train_acc:.3f}, valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}"
                # )
            else:
                cnt += 1
                if cnt == self.estop_steps:
                    print(
                        f"{graph.name} Early Stopping! Best Epoch: {best_epoch}, best val acc: {best_acc:.3f}, test acc: {best_acc_t:.3f}"
                    )
                    break
        t_finish = time.time()
        print("\n10 epoch cost: {:.4f}s\n".format((t_finish - t_start) / (epoch + 1) * 10))
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
        print(self.gates)
        self.best_epoch = best_epoch

    def joint_learn(
        self,
        graph,
        labels,
        train_mask,
        val_mask,
        test_mask,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.n_edges = graph.num_edges()
        self.n_nodes = graph.num_nodes()
        self.get_neighborhood_features(graph)
        self.to(self.device)
        labels = labels.to(device)
        best_epoch = 0
        best_acc = 0.0
        cnt = 0
        best_state_dict = None

        t_start = time.time()
        for epoch in range(1000):
            self.train()
            self.optimizer.zero_grad()

            embeddings = self.forward(self.features, frozen_experts=False)
            logits_onehot = self.classifier(embeddings)
            loss = self.ce(logits_onehot[train_mask], labels[train_mask])

            loss.backward()
            self.optimizer.step()

            [train_acc, valid_acc, test_acc] = self.test(
                self.features, labels, [train_mask, val_mask, test_mask]
            )
            print(
                f"epoch:{epoch},train acc: {train_acc:.3f}, valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}"
            )
            if valid_acc > best_acc:
                cnt = 0
                best_acc = valid_acc
                best_acc_t = test_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
                # print(f"\nEpoch:{epoch}, Loss:{loss.item()}")
                # print(
                #     f"train acc: {train_acc:.3f}, valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}"
                # )
            else:
                cnt += 1
                if cnt == self.estop_steps:
                    print(
                        f"{graph.name} Early Stopping! Best Epoch: {best_epoch}, best val acc: {best_acc:.3f}, test acc: {best_acc_t:.3f}"
                    )
                    break
        t_finish = time.time()
        print("\n10 epoch cost: {:.4f}s\n".format((t_finish - t_start) / (epoch + 1) * 10))
        self.load_state_dict(best_state_dict)
        print(self.gates)
        self.best_epoch = best_epoch

    def fit(
        self,
        graph,
        labels,
        train_mask,
        val_mask,
        test_mask,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.pretrain_experts(graph, device)
        self.pretrain_gating(
            graph,
            labels,
            train_mask,
            val_mask,
            test_mask,
            device=device,
        )
        self.joint_learn(
            graph,
            labels,
            train_mask,
            val_mask,
            test_mask,
            device=device,
        )


def loss_function(adj_preds, adj_labels):
    """compute loss

    Args:
        adj_preds (torch.Tensor):reconstructed adj

    Returns:
        torch.Tensor: loss
    """

    cost = 0.0
    cost += F.binary_cross_entropy_with_logits(adj_preds, adj_labels)

    return cost


def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num):
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
    f_adj = np.matmul(z, np.transpose(z))
    cosine = f_adj
    cosine = cosine.reshape(
        [
            -1,
        ]
    )
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1 - lower_treshold) * len(cosine))

    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    del cosine
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


#
