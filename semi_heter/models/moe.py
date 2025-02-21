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


class ONGNNConv(nn.Module):
    def __init__(self, tm_net, tm_norm, simple_gating, tm, diff_or, repeats):
        super(ONGNNConv, self).__init__()
        self.tm_net = tm_net
        self.tm_norm = tm_norm
        self.simple_gating = simple_gating
        self.tm = tm
        self.diff_or = diff_or
        self.repeats = repeats

    def forward(self, x, m, last_tm_signal):
        if self.tm == True:
            if self.simple_gating == True:
                tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
            else:
                tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
                tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
                if self.diff_or == True:
                    tm_signal_raw = last_tm_signal + (1 - last_tm_signal) * tm_signal_raw
            tm_signal = tm_signal_raw.repeat_interleave(repeats=self.repeats, dim=1)
            out = x * tm_signal + m * (1 - tm_signal)
        else:
            out = m
            tm_signal_raw = last_tm_signal

        out = self.tm_norm(out)

        return out, tm_signal_raw


class MoE(nn.Module):
    """MoE"""

    def __init__(
        self,
        in_feats,
        h_feats,
        n_nodes,
        # NOTE: neighborhood_order >= 1 as ego node is the 1st order of the neighborhood
        neighborhood_order,
        n_gnn_layers: int,
        n_agnn_layers: int,
        n_clusters,
        n_exp_epochs: int = 100,
        n_jl_epochs: int = 2000,
        lr: float = 0.001,
        l2_coef: float = 0.0001,
        early_stop: int = 100,
        device=None,
        act=nn.ReLU,
        normalize=True,
        alpha=0.5,
        high_ratio=1,
        low_ratio=1,
        n_experts=5,
        gating="emb",
        n_gating=1,
        noisy_gating=True,
        norm: str = "sym",
        renorm: bool = True,
        upth_st: float = 0.0015,
        upth_ed: float = 0.001,
        lowth_st: float = 0.1,
        lowth_ed: float = 0.5,
        upd: float = 10,
        bs: int = 1024,
        dropout: float = 0.5,
        dropout2: float = 0.9,
        k=2,
        br: float = 0.01,
        n_hops=6,
        n_intervals=3,
    ) -> None:
        super().__init__()
        self.n_intervals = n_intervals

        # adaptive learning
        self.upth_st = upth_st
        self.upth_ed = upth_ed
        self.lowth_st = lowth_st
        self.lowth_ed = lowth_ed
        self.upd = upd
        self.bs = bs
        self.norm = norm
        self.renorm = renorm

        self.n_jl_epochs = n_jl_epochs
        self.dropout = dropout
        self.dropout2 = dropout2
        self.br = br

        # MoE
        self.noisy_gating = noisy_gating
        self.gating = gating
        self.loss_coef = alpha
        self.h_feats = h_feats
        self.l2_coef = l2_coef

        self.n_gating = n_gating
        self.n_experts = n_experts

        # instantiate experts
        if self.gating == "emb":
            # feats = h_feats * self.n_experts
            feats = h_feats * (n_hops + 1)
            self.w_gate = nn.Parameter(torch.zeros(feats, self.n_experts), requires_grad=True)
            # self.w_gate_1 = nn.Parameter(torch.zeros(2048, self.n_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(feats, self.n_experts), requires_grad=True)
            # self.w_gate_disjoint = nn.Parameter(torch.zeros(2 * h_feats, 2), requires_grad=True)
            # self.w_noise_disjoint = nn.Parameter(torch.zeros(2 * h_feats, 2), requires_grad=True)
        else:
            self.w_gate = nn.Parameter(torch.zeros(in_feats, self.n_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(in_feats, self.n_experts), requires_grad=True)

        # num_layers = 4
        # hidden_channel=h_feats
        # self.chunk_size = 128
        # global_gating = False
        # num_layers_input=1
        # tm=True
        # diff_or=True

        # self.tm_norm = ModuleList()
        # self.tm_net = ModuleList()
        # self.moe = ModuleList()
        # self.mlp = SelfReconstruction(
        #     in_feats=in_feats,
        #     h_feats=h_feats,
        #     dropout=dropout,
        # )
        # # self.linear_trans_in.append(Linear(in_feats, hidden_channel))
        # # self.norm_input.append(LayerNorm(hidden_channel))

        # # for i in range(num_layers_input - 1):
        # #     self.linear_trans_in.append(Linear(hidden_channel, hidden_channel))
        # #     self.norm_input.append(LayerNorm(hidden_channel))

        # if global_gating == True:
        #     # tm_net = Linear(2 * hidden_channel, self.chunk_size)
        #     tm_net = MLP(in_feats=2 * hidden_channel, h_feats=[self.chunk_size], layers=1, acts=[])
        # for i in range(num_layers):
        #     self.tm_norm.append(LayerNorm(hidden_channel))
        #     if global_gating == False:
        #         # self.tm_net.append(Linear(2 * hidden_channel, self.chunk_size))
        #         self.tm_net.append(MLP(in_feats=2 * hidden_channel, h_feats=[self.chunk_size], layers=1, acts=[]))
        #     else:
        #         self.tm_net.append(tm_net)
        #     self.moe.append(
        #         ONGNNConv(
        #             tm_net=self.tm_net[i],
        #             tm_norm=self.tm_norm[i],
        #             simple_gating=False,
        #             tm=tm,
        #             diff_or=diff_or,
        #             repeats=int(hidden_channel / self.chunk_size),
        #         )
        #     )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        # assert(self.n_gating <= self.n_experts)

        # URL
        self.lr = lr
        self.n_exp_epochs = n_exp_epochs
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

        # self.attr_hom_exp = AdaptiveLearning(
        #     in_feats=in_feats,
        #     h_feats=h_feats,
        #     n_gnn_layers=n_agnn_layers,
        #     norm=norm,
        #     renorm=renorm,
        #     upth_st=upth_st,
        #     upth_ed=upth_ed,
        #     lowth_st=lowth_st,
        #     lowth_ed=lowth_ed,
        #     upd=upd,
        #     dropout=dropout,
        # )

        # self.attr_heter_exp = SelfReconstruction(
        #     in_feats=in_feats,
        #     h_feats=h_feats,
        #     dropout=dropout,
        # )

        # self.adj_heter_exp = SelfReconstruction(
        #     in_feats=n_nodes,
        #     h_feats=h_feats,
        #     dropout=0.0,
        # )

        # self.stru_heter_exp = NeighborhoodPrediction(
        #     in_feats=in_feats,
        #     h_feats=h_feats,
        #     neighborhood_order=neighborhood_order,
        #     alpha=alpha,
        #     device=device,
        #     n_clusters=n_clusters,
        #     dropout=dropout,
        #     k=k,
        # )

        # self.stru_hom_exp = EdgeReconstruction(
        #     in_feats=in_feats,
        #     h_feats=h_feats,
        #     n_gnn_layers=n_gnn_layers,
        #     dropout=dropout,
        # )

        from .SetGNN import SetGNN

        self.set_gnn = SetGNN(
            in_feats=in_feats,
            h_feats=h_feats,
            n_hops=n_hops,
            dropout=dropout,
            n_intervals=n_intervals,
        )
        self.set_gnn_1 = None
        # self.set_gnn_1 = SetGNN(
        #     in_feats=h_feats * (n_hops - n_intervals + 2),
        #     h_feats=h_feats,
        #     n_hops=n_hops,
        #     dropout=dropout,
        #     n_intervals=n_intervals,
        #     no_save=True,
        # )

        # self.attr_hom_exp = SetGNN(
        #     in_feats=in_feats,
        #     h_feats=h_feats,
        #     n_hops=3,
        #     dropout=dropout,
        # )
        # layers = 1
        # hs = [h_feats]
        # n = int(math.log2(h_feats * (n_hops + 1))) - math.log2(h_feats)
        # while n // 2 > 0:
        #     layers += 1
        #     n = n // 2
        #     hs.append(hs[-1] * 2)
        # hs = hs[::-1]
        # hs.append(n_clusters)
        # acts = [nn.ReLU()] * n_hops
        # acts.append( nn.Softmax(dim=1))
        # from ..modules.mlp import Res
        # self.classifier = MLP(
        #     in_feats=h_feats * (n_hops + 1),
        #     h_feats=hs,
        #     layers=layers + 1,
        #     acts=acts,
        # )
        self.classifier = MLP(
            in_feats=max(h_feats * (n_hops - n_intervals + 2), h_feats),
            # in_feats=h_feats * (n_hops +1),
            h_feats=[n_clusters],
            layers=1,
            # acts=[nn.Softmax(dim=1)],
            acts=[nn.Identity()],
            dropout=self.dropout2,
        )

        self.ce = torch.nn.CrossEntropyLoss()

        self.warmup_prepared = False
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.l2_coef,
        )

    def forward(
        self,
        features,
        graph=None,
        nodes_batch=None,
        frozen_experts=False,
        device=None,
    ):
        z_attr_hom_exp = None
        z_attr_heter_exp = None
        z_stru_heter_exp = None
        z_stru_hom_exp = None
        z_adj_heter_exp = None

        if frozen_experts:
            with torch.no_grad():
                # z_attr_hom_exp = self.attr_hom_exp(
                #     self.attr_hom_exp.sm_fea_s
                #     if nodes_batch is None
                #     else self.attr_hom_exp.sm_fea_s[nodes_batch]
                # )
                # z_attr_hom_exp = self.attr_hom_exp(
                #    features,graph=graph,device=device,k_kernel=3,
                # )
                # z_stru_hom_exp = self.stru_hom_exp(
                #     self.stru_hom_exp.sm_fea_s
                #     if nodes_batch is None
                #     else self.stru_hom_exp.sm_fea_s[nodes_batch]
                # )
                # z_attr_heter_exp = self.attr_heter_exp(
                #     features if nodes_batch is None else features[nodes_batch]
                # )
                # z_stru_heter_exp = (
                #     self.stru_heter_exp(features)
                #     if nodes_batch is None
                #     else self.stru_heter_exp(features)[nodes_batch]
                # )
                # z_adj_heter_exp = self.adj_heter_exp(
                #     self.adj_sparse if nodes_batch is None else self.adj_sparse[nodes_batch]
                # )
                z_set_gnn = self.set_gnn(graph=graph, device=device)
                z_set_gnn = (
                    self.set_gnn_1(graph=graph, device=device, feats=torch.cat(z_set_gnn, dim=1))
                    if self.set_gnn_1 is not None
                    else z_set_gnn
                )

        else:
            # z_attr_hom_exp = self.attr_hom_exp(
            #     self.attr_hom_exp.sm_fea_s
            #     if nodes_batch is None
            #     else self.attr_hom_exp.sm_fea_s[nodes_batch]
            # )
            # z_attr_hom_exp = self.attr_hom_exp(
            #     features,graph=graph,device=device,k_kernel=3,
            # )
            # z_stru_hom_exp = self.stru_hom_exp(
            #     self.stru_hom_exp.sm_fea_s
            #     if nodes_batch is None
            #     else self.stru_hom_exp.sm_fea_s[nodes_batch]
            # )
            # z_attr_heter_exp = self.attr_heter_exp(
            #     features if nodes_batch is None else features[nodes_batch]
            # )
            # z_stru_heter_exp = (
            #     self.stru_heter_exp(features)
            #     if nodes_batch is None
            #     else self.stru_heter_exp(features)[nodes_batch]
            # )
            # z_adj_heter_exp = self.adj_heter_exp(
            #     self.adj_sparse if nodes_batch is None else self.adj_sparse[nodes_batch]
            # )
            z_set_gnn = self.set_gnn(graph=graph, device=device)
            z_set_gnn = (
                self.set_gnn_1(graph=graph, device=device, feats=torch.cat(z_set_gnn, dim=1))
                if self.set_gnn_1 is not None
                else z_set_gnn
            )
        # z_set_gnn.extend([
        #             # self.stru_heter_exp.fsd[0],
        #             # self.stru_heter_exp.fsd[1],
        #             # self.stru_heter_exp.fsm[1],
        #             # z_attr_hom_exp,
        #             z_attr_heter_exp,
        #             # z_stru_heter_exp,
        #             # z_stru_hom_exp,
        #             # z_adj_heter_exp,
        #         ])
        # z_set_gnn.extend(z_attr_hom_exp)
        embs_list = [
            # self.stru_heter_exp.fsd[0],
            # self.stru_heter_exp.fsd[1],
            # self.stru_heter_exp.fsm[1],
            z_attr_hom_exp,
            z_attr_heter_exp,
            z_stru_heter_exp,
            z_stru_hom_exp,
            z_adj_heter_exp,
        ]
        # if self.gating == "emb":
        #     self.gates, self.load = gates, load = self.noisy_top_k_gating(
        #         F.dropout(
        #             torch.cat(
        #                 z_set_gnn,
        #                 # [
        #                 #     # self.stru_heter_exp.fsd[0],
        #                 #     # self.stru_heter_exp.fsd[1],
        #                 #     # self.stru_heter_exp.fsm[1],
        #                 #     z_attr_hom_exp,
        #                 #     z_attr_heter_exp,
        #                 #     z_stru_heter_exp,
        #                 #     z_stru_hom_exp,
        #                 #     z_adj_heter_exp,
        #                 # ],
        #                 dim=1,
        #             ),
        #             self.dropout,
        #             training=self.training,
        #         ),
        #         w_gate=self.w_gate,
        #         w_noise=self.w_noise,
        #         n_gating=self.n_gating,
        #         train=self.training,
        #     )

        # else:
        #     self.gates, self.load = gates, load = self.noisy_top_k_gating(
        #         F.dropout(
        #             features,
        #             self.dropout,
        #             training=self.training,
        #         ),
        #         w_gate=self.w_gate,
        #         w_noise=self.w_noise,
        #         n_gating=self.n_gating,
        #         train=self.training,
        #     )
        return torch.cat(z_set_gnn, dim=1), embs_list
        # return (
        #     torch.matmul(
        #         torch.stack(
        #             z_set_gnn,
        #             # [
        #             #     # self.stru_heter_exp.fsd[0],
        #             #     # self.stru_heter_exp.fsd[1],
        #             #     # self.stru_heter_exp.fsm[1],
        #             #     z_attr_hom_exp,
        #             #     z_attr_heter_exp,
        #             #     z_stru_heter_exp,
        #             #     z_stru_hom_exp,
        #             #     z_adj_heter_exp,
        #             # ],
        #             dim=2,
        #         ),
        #         gates.unsqueeze(dim=2),
        #     ).squeeze(dim=2),
        #     embs_list,
        # )

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values, n_gating):
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

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + n_gating
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

    def noisy_top_k_gating(
        self,
        x,
        w_gate,
        w_noise,
        n_gating,
        train,
        noise_epsilon=1e-2,
    ):
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
        clean_logits = x @ w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(n_gating + 1, self.n_experts), dim=1)
        top_k_logits = top_logits[:, :n_gating]
        top_k_indices = top_indices[:, :n_gating]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and n_gating < self.n_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits,
                    noisy_logits,
                    noise_stddev,
                    top_logits,
                    n_gating=n_gating,
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def warmup_experts(
        self,
        graph,
        batch_size,
        load_state=False,
        device: torch.device = torch.device("cpu"),
    ):
        if not self.warmup_prepared:
            self.device = device
            self.to(self.device)

            attr_hom_dropout, self.attr_hom_exp.dropout = self.attr_hom_exp.dropout, 0
            attr_heter_dropout, self.attr_heter_exp.dropout = self.attr_heter_exp.dropout, 0
            stru_hom_dropout, self.stru_hom_exp.dropout = self.stru_hom_exp.dropout, 0
            stru_heter_dropout, self.stru_heter_exp.ec.dropout = self.stru_heter_exp.ec.dropout, 0

            self.attr_hom_exp.fit(
                graph=graph,
                lr=self.lr,
                n_epochs=self.n_exp_epochs,
                batch_size=batch_size,
                load_state=load_state,
                state=f"tmp/experts_v{2.1}/{self.attr_hom_exp._get_name()}_{graph.name}_h_{self.h_feats}.pt",
                device=device,
            )
            self.attr_heter_exp.fit(
                graph=graph,
                lr=self.lr,
                n_epochs=self.n_exp_epochs,
                batch_size=batch_size,
                load_state=load_state,
                state=f"tmp/experts_v{2.1}/{self.attr_heter_exp._get_name()}_{graph.name}_h_{self.h_feats}.pt",
                device=device,
            )
            self.stru_hom_exp.fit(
                graph=graph,
                lr=self.lr,
                n_epochs=self.n_exp_epochs,
                batch_size=batch_size,
                load_state=load_state,
                state=f"tmp/experts_v{2.1}/{self.stru_hom_exp._get_name()}_{graph.name}_h_{self.h_feats}.pt",
                device=device,
            )
            self.stru_heter_exp.fit(
                graph=graph,
                lr=self.lr,
                n_epochs=self.n_exp_epochs,
                batch_size=batch_size,
                load_state=load_state,
                state=f"tmp/experts_v{2.1}/{self.stru_heter_exp._get_name()}_{graph.name}_h_{self.h_feats}.pt",
                device=device,
            )

            self.attr_hom_exp.dropout = attr_hom_dropout
            self.attr_heter_exp.dropout = attr_heter_dropout
            self.stru_hom_exp.dropout = stru_hom_dropout
            self.stru_heter_exp.ec.dropout = stru_heter_dropout

            self.warmup_prepared = True

    def validate(self, features, labels, train_mask, val_mask, test_mask, graph):
        self.train(False)
        with torch.no_grad():
            (
                embeddings,
                _,
            ) = self.forward(
                features,
                frozen_experts=True,
                graph=graph,
                device=self.device,
            )
            logits_onehot = self.classifier(embeddings)
            y_pred = torch.argmax(logits_onehot, dim=1)
        return (
            ACC(labels[train_mask].cpu(), y_pred[train_mask].cpu()),
            ACC(labels[val_mask].cpu(), y_pred[val_mask].cpu()),
            ACC(labels[test_mask].cpu(), y_pred[test_mask].cpu()),
        )

    def warmup_gating(
        self,
        graph,
        labels,
        train_mask,
        val_mask,
        test_mask,
        bs=2048,
        split_id=None,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.to(self.device)
        labels = labels.to(device)

        best_epoch = 0
        best_acc = 0.0
        cnt = 0
        best_state_dict = None
        writer = SummaryWriter(
            log_dir=f"logs/runs/{get_str_time()[:10]}/warmup_{graph.name}_{split_id}_{self.h_feats}_{self.n_gating}_{get_str_time()[11:]}"
        )

        self.init_joint_experts(
            graph=graph,
            device=device,
            bs=bs,
            n_epochs=self.n_jl_epochs,
        )
        t_start = time.time()
        for epoch in range(self.n_jl_epochs):
            t = time.time()
            self.train()
            self.optimizer.zero_grad()

            (
                embeddings,
                _,
            ) = self.forward(
                graph.ndata["feat"],
                frozen_experts=True,
                device=device,
            )
            logits_onehot = self.classifier(embeddings)
            loss = self.ce(logits_onehot[train_mask], labels[train_mask])
            loss_val = self.ce(logits_onehot[val_mask], labels[val_mask])
            loss_test = self.ce(logits_onehot[test_mask], labels[test_mask])

            loss.backward()
            self.optimizer.step()

            (train_acc, valid_acc, test_acc) = self.validate(
                graph.ndata["feat"],
                labels,
                train_mask,
                val_mask,
                test_mask,
                graph=graph,
            )
            if valid_acc > best_acc:
                cnt = 0
                best_acc = valid_acc
                best_acc_t = test_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
            else:
                cnt += 1
                if cnt == self.estop_steps:
                    print(
                        f"{graph.name} Early Stopping! Best Epoch: {best_epoch}, best val acc: {best_acc:.3f}, test acc: {best_acc_t:.3f}"
                    )
                    break

            writer.add_scalar("warmup/time/train", time.time() - t, epoch)

            writer.add_scalar("warmup/metric/train", train_acc, epoch)
            writer.add_scalar("warmup/metric/val", valid_acc, epoch)
            writer.add_scalar("warmup/metric/test", test_acc, epoch)
            writer.add_scalar("warmup/loss/train", loss.item(), epoch)
            writer.add_scalar("warmup/loss/val", loss_val.item(), epoch)
            writer.add_scalar("warmup/loss/test", loss_test.item(), epoch)
        t_finish = time.time()
        print(f"10 epoch cost: {(t_finish-t_start)/self.n_jl_epochs * 10:.4f}s")
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

    def init_joint_experts(
        self,
        graph,
        n_epochs,
        bs,
        device: torch.device = torch.device("cpu"),
    ):
        self.stru_hom_exp.init_batch(
            graph=graph,
            device=device,
        )
        self.stru_heter_exp.init_batch(
            graph=graph,
            device=device,
        )
        (
            self.attr_hom_exp.bs,
            self.attr_hom_exp.pos_inds,
            self.attr_hom_exp.neg_inds,
            self.attr_hom_exp.upth,
            self.attr_hom_exp.lowth,
        ) = self.attr_hom_exp.init_batch(
            graph=graph,
            device=device,
            n_epochs=n_epochs,
            bs=bs,
        )

    def forward_joint_experts(
        self,
        graph,
        z_attr_heter_exp,
        z_stru_hom_exp,
        z_stru_heter_exp,
        bs,
        n_epochs,
        epoch,
        device: torch.device = torch.device("cpu"),
    ):
        loss_attr_hom_exp = self.attr_hom_exp.full_batch_forward(
            graph=graph,
            bs=bs,
            n_epochs=n_epochs,
            epoch=epoch,
            device=device,
        )
        # loss_attr_hom_exp.backward()
        # self.optimizer.step()

        loss_attr_heter_exp = self.attr_heter_exp.full_batch_forward(
            graph=graph,
            z=z_attr_heter_exp,
            device=device,
        )
        # loss_attr_heter_exp.backward()
        # self.optimizer.step()

        loss_stru_hom_exp = self.stru_hom_exp.full_batch_forward(
            graph=graph,
            z=z_stru_hom_exp,
            device=device,
        )
        # loss_stru_hom_exp.backward()
        # self.optimizer.step()

        loss_stru_heter_exp = self.stru_heter_exp.full_batch_forward(
            graph=graph,
            z=z_stru_heter_exp,
            device=device,
        )
        # loss_stru_heter_exp.backward()
        # self.optimizer.step()

        return (
            loss_attr_hom_exp,
            loss_attr_heter_exp,
            loss_stru_hom_exp,
            loss_stru_heter_exp,
        )

    def joint_learn(
        self,
        graph,
        labels,
        train_mask,
        val_mask,
        test_mask,
        bs=2048,
        split_id=None,
        joint_experts_loss=True,
        device: torch.device = torch.device("cpu"),
    ):

        self.device = device
        self.to(self.device)
        labels = labels.to(device)

        best_epoch = 0
        best_acc = 0.0
        cnt = 0
        best_state_dict = None
        writer = SummaryWriter(
            log_dir=f"logs/runs/{get_str_time()[:10]}/joint_{graph.name}_{split_id}_{self.h_feats}_{self.n_gating}_{get_str_time()[11:]}"
        )

        # self.init_joint_experts(
        #     graph=graph,
        #     device=device,
        #     n_epochs=self.n_exp_epochs,
        #     bs=bs,
        # )

        t_start = time.time()
        self.adj_sparse = sparse_mx_to_torch_sparse_tensor(graph.adj_external(scipy_fmt="csr")).to(
            device
        )
        for epoch in range(self.n_jl_epochs):
            t = time.time()
            self.train()
            self.optimizer.zero_grad()

            if joint_experts_loss:
                st = 0
                ed = self.attr_hom_exp.bs
                cur_loss = 0.0

                length = len(self.attr_hom_exp.pos_inds)
                while ed <= length:
                    xind, yind = self.attr_hom_exp.sample_batch(
                        st=st,
                        ed=ed,
                        neg_inds=self.attr_hom_exp.neg_inds,
                        pos_inds=self.attr_hom_exp.pos_inds,
                        n_nodes=graph.num_nodes(),
                        device=device,
                    )
                    nodes_batch = torch.LongTensor(torch.cat((xind, yind)).cpu())

                    (
                        embeddings,
                        [
                            z_attr_hom_exp,
                            z_attr_heter_exp,
                            z_stru_hom_exp,
                            z_stru_heter_exp,
                        ],
                    ) = self.forward(
                        graph.ndata["feat"],
                        nodes_batch=nodes_batch,
                        frozen_experts=False,
                        device=device,
                    )
                    logits_onehot = self.classifier(embeddings)

                    loss = self.ce(
                        logits_onehot[train_mask[nodes_batch]],
                        labels[nodes_batch][train_mask[nodes_batch]],
                    )
                    loss_val = self.ce(
                        logits_onehot[val_mask[nodes_batch]],
                        labels[nodes_batch][val_mask[nodes_batch]],
                    )
                    loss_test = self.ce(
                        logits_onehot[test_mask[nodes_batch]],
                        labels[nodes_batch][test_mask[nodes_batch]],
                    )

                    loss_attr_hom_exp = self.attr_hom_exp.batch_forward(
                        graph=graph,
                        x_batch=xind,
                        y_batch=yind,
                        device=device,
                        z=z_attr_hom_exp,
                    )
                    loss_stru_hom_exp = self.stru_hom_exp.batch_forward(
                        graph=graph,
                        nodes_batch=nodes_batch,
                        device=device,
                        z=z_stru_hom_exp,
                    )
                    loss_attr_heter_exp = self.attr_heter_exp.batch_forward(
                        graph=graph,
                        nodes_batch=nodes_batch,
                        device=device,
                        z=z_attr_heter_exp,
                    )
                    loss_stru_heter_exp = self.stru_heter_exp.batch_forward(
                        graph=graph,
                        nodes_batch=nodes_batch,
                        device=device,
                        z=z_stru_heter_exp,
                    )
                    # loss_cv =self.cv_squared(self.gates.sum(0))
                    # loss_cv = self.cv_squared(self.gates.sum(0)) + self.cv_squared(self.load)
                    loss_cv = 0
                    loss = (
                        loss
                        + 0.01 * loss_attr_hom_exp
                        + 0.01 * loss_attr_heter_exp
                        + 0.01 * loss_stru_hom_exp
                        + 0.01 * loss_stru_heter_exp
                        + self.br * loss_cv
                    )
                    loss.backward()
                    self.optimizer.step()
                    cur_loss += loss.item()

                    st = ed
                    if ed < length <= ed + self.attr_hom_exp.bs:
                        ed += length - ed
                    else:
                        ed += self.attr_hom_exp.bs

                (
                    self.attr_hom_exp.pos_inds,
                    self.attr_hom_exp.neg_inds,
                    self.attr_hom_exp.bs,
                    self.attr_hom_exp.upth,
                    self.attr_hom_exp.lowth,
                ) = self.attr_hom_exp.update_batch(
                    epoch,
                    graph,
                    self.attr_hom_exp.pos_inds,
                    self.attr_hom_exp.neg_inds,
                    self.attr_hom_exp.bs,
                    self.attr_hom_exp.upth,
                    self.attr_hom_exp.lowth,
                )
            else:
                (
                    embeddings,
                    [
                        z_attr_hom_exp,
                        z_attr_heter_exp,
                        z_stru_hom_exp,
                        z_stru_heter_exp,
                        z_adj_heter_exp,
                    ],
                ) = self.forward(
                    graph.ndata["feat"],
                    frozen_experts=False,
                    graph=graph,
                    device=device,
                )
                logits_onehot = self.classifier(embeddings)
                # if (epoch + 1) % 10 == 0:
                #     self.attr_hom_exp.upth, self.attr_hom_exp.lowth = update_threshold(
                #         self.attr_hom_exp.upth_st,
                #         self.attr_hom_exp.lowth_st,
                #         self.attr_hom_exp.up_eta,
                #         self.attr_hom_exp.low_eta,
                #     )
                #     self.attr_hom_exp.pos_inds, self.attr_hom_exp.neg_inds = update_similarity(
                #         embeddings.detach().cpu().numpy(),
                #         self.upth_st,
                #         self.lowth_st,
                #     )
                #     self.attr_hom_exp.generate_graph(
                #         pos_inds=self.attr_hom_exp.pos_inds,
                #         n_nodes=graph.num_nodes(),
                #         x=graph.ndata["feat"],
                #         device=graph.device,
                #     )
                # loss_cv = self.cv_squared(self.gates.sum(0))
                # loss_cv = 0
                # loss_cv = self.cv_squared(self.gates.sum(0)) + self.cv_squared(self.load)
                loss_cv = 0
                loss = self.ce(logits_onehot[train_mask], labels[train_mask])
                loss_val = self.ce(logits_onehot[val_mask], labels[val_mask])
                loss_test = self.ce(logits_onehot[test_mask], labels[test_mask])
                (loss + self.br * loss_cv).backward()
                self.optimizer.step()
                loss_attr_hom_exp = loss_attr_heter_exp = loss_stru_hom_exp = (
                    loss_stru_heter_exp
                ) = torch.tensor(0)

            (train_acc, valid_acc, test_acc) = self.validate(
                graph.ndata["feat"],
                labels,
                train_mask,
                val_mask,
                test_mask,
                graph=graph,
            )
            print(
                f"epoch:{epoch},loss: {loss.item()}, train acc: {train_acc:.3f}, valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}"
            )
            if valid_acc >= best_acc:
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

            writer.add_scalar("joint/time/train", time.time() - t, epoch)

            writer.add_scalar("joint/metric/train", train_acc, epoch)
            writer.add_scalar("joint/metric/val", valid_acc, epoch)
            writer.add_scalar("joint/metric/test", test_acc, epoch)
            writer.add_scalar("joint/loss/train", loss.item(), epoch)
            writer.add_scalar("joint/loss/val", loss_val.item(), epoch)
            writer.add_scalar("joint/loss/test", loss_test.item(), epoch)
            writer.add_scalar("joint/loss/loss_attr_hom_exp", loss_attr_hom_exp.item(), epoch)
            writer.add_scalar("joint/loss/loss_attr_heter_exp", loss_attr_heter_exp.item(), epoch)
            writer.add_scalar("joint/loss/loss_stru_hom_exp", loss_stru_hom_exp.item(), epoch)
            writer.add_scalar("joint/loss/loss_stru_heter_exp", loss_stru_heter_exp.item(), epoch)
        t_finish = time.time()
        print(f"10 epoch cost: {(t_finish-t_start)/self.n_jl_epochs * 10:.4f}s")
        # with torch.no_grad():
        #     print(
        #         "gates:",
        #         self.noisy_top_k_gating(
        #             F.dropout(
        #                 torch.concat(
        #                     [
        #                         z_attr_hom_exp,
        #                         z_attr_heter_exp,
        #                         z_stru_heter_exp,
        #                         z_stru_hom_exp,
        #                         z_adj_heter_exp,
        #                     ],
        #                     dim=1,
        #                 ),
        #                 self.dropout,
        #                 training=False,
        #             ),
        #             w_gate=self.w_gate,
        #             w_noise=self.w_noise,
        #             n_gating=self.n_gating,
        #             train=self.training,
        #         ),
        #     )
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

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

    def get_embeddings(self):
        with torch.no_grad():
            pass
