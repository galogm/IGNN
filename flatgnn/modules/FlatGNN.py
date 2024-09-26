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
from dgl.nn.pytorch import GraphConv
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
from torch_geometric.nn import GATConv

from ..utils import preprocess_neighborhoods, metric
from .DeepSets import DeepSets
from .LSTM import LSTM
from .MLP import MLP
from .orderedGating import ONGNNConv


acts = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "sigmoid": nn.Sigmoid,
    "none": nn.Identity,
}


class IGFormer(nn.Module):
    def __init__(
        self,
        num_heads,
        in_ndim_trans,
        out_ndim_trans=64,
        trans_dropout=0.2,
        act="none",
    ):
        super().__init__()
        self.num_heads = num_heads

        self.in_ndim_trans = in_ndim_trans
        # self.in_ndim_trans = h_feats
        self.out_ndim_trans = out_ndim_trans

        self.trans_dropout = trans_dropout

        self.proj = (
            MLP(
                in_feats=self.in_ndim_trans,
                h_feats=[self.out_ndim_trans * self.num_heads],
                acts=[acts[act]()],
                dropout=trans_dropout,
                layer_norm=True,
            )
            if self.in_ndim_trans != self.out_ndim_trans * self.num_heads
            else nn.Identity()
        )

        self.fc = MLP(
            in_feats=self.out_ndim_trans * self.num_heads,
            h_feats=[self.out_ndim_trans * self.num_heads],
            acts=[acts[act]()],
            dropout=trans_dropout,
            layer_norm=True,
        )

        # self.w_q = MLP(
        #     in_feats=self.in_ndim_trans,
        #     h_feats=[self.out_ndim_trans * self.num_heads],
        #     acts=[acts[act]()],
        #     dropout=trans_dropout,
        #     layer_norm=False,
        # )
        self.w_q = MLP(
            in_feats=self.in_ndim_trans,
            h_feats=[self.out_ndim_trans * self.num_heads],
            acts=[acts["none"]()],
            dropout=trans_dropout,
            layer_norm=False,
        )
        self.w_k = MLP(
            in_feats=self.in_ndim_trans,
            h_feats=[self.out_ndim_trans * self.num_heads],
            acts=[acts["none"]()],
            dropout=trans_dropout,
            layer_norm=False,
        )
        self.w_v = MLP(
            in_feats=self.in_ndim_trans,
            h_feats=[self.out_ndim_trans * self.num_heads],
            # h_feats=[self.h_feats * self.num_heads],
            acts=[acts["none"]()],
            dropout=trans_dropout,
            layer_norm=False,
        )
        self.h_lins = torch.nn.Linear(self.in_ndim_trans, self.out_ndim_trans * self.num_heads)
        self.layer_n = torch.nn.LayerNorm(self.out_ndim_trans * self.num_heads)
        self.lin_out = torch.nn.Linear(self.out_ndim_trans * self.num_heads, self.out_ndim_trans * self.num_heads)
        self.beta = torch.nn.Parameter(torch.zeros(self.out_ndim_trans * self.num_heads))

    # def full_attention_conv(self, qs, ks, vs, h=None, output_attn=False):
    #     # normalize input
    #     # qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    #     # ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    #     n, d, h = qs.size()
    #     N = qs.shape[0]

    #     # numerator
    #     kvs = torch.einsum("ndh, nmh -> dmh", ks, vs)
    #     attention_num = torch.einsum("ndh, dmh -> nmh", qs, kvs)  # [N, H, D]
    #     # attention_num += N * vs

    #     # denominator
    #     # all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    #     # ks_sum = torch.einsum("ndh,n->dh", ks, all_ones)
    #     ks_sum = torch.einsum("ndh->dh", ks)
    #     # ks_sum = torch.einsum("ndh->dh", ks, all_ones)
    #     attention_normalizer = torch.einsum("ndh,dh->nh", qs, ks_sum).unsqueeze(1)  # [N, H]

    #     # attentive aggregated results
    #     # attention_normalizer = torch.unsqueeze(
    #     #     attention_normalizer, len(attention_normalizer.shape)
    #     # )  # [N, H, 1]
    #     # attention_normalizer += torch.ones_like(attention_normalizer) * N
    #     attn_output = attention_num / attention_normalizer  # [N, H, D]

    #     att = attn_output
    #     # print(att.shape, attn_output.shape)
    #     return F.dropout(
    #         att * vs,
    #         p=self.trans_dropout,
    #         training=self.training,
    #     )
    def full_attention_conv_sg(self, qs, ks, vs, output_attn=False):
        seq_len, n_hidden, n_heads = qs.size()
        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape)
        )  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        if output_attn:
            return attn_output, attention
        else:
            return attn_output.reshape(seq_len, -1)

    def full_attention_conv_poly(self, qs, ks, vs, x, output_attn=False):
        seq_len, n_hidden, n_heads = qs.size()

        h = self.h_lins(x)
        k = F.sigmoid(ks)
        q = F.sigmoid(qs)
        v = vs

        # numerator
        kv = torch.einsum('nhd, nhm -> dhm', k, v)
        num = torch.einsum('nhd, dhm -> nhm', q, kv)

        # denominator
        k_sum = torch.einsum('nhd -> hd', k)
        den = torch.einsum('nhd, hd -> nh', q, k_sum).unsqueeze(-1)

        # linear global attention based on kernel trick
        beta = F.sigmoid(self.beta).unsqueeze(0)
        x = (num/den).reshape(seq_len, -1)
        x = self.layer_n(x) * (h+beta)
        x = F.relu(self.lin_out(x))
        # x = F.dropout(x, p=0.2, training=self.training)

        return x

    def forward(
        self,
        h_trans,
    ):
        h_trans = F.layer_norm(h_trans, h_trans.shape)
        # Apply a linear transformation (optional) before self-attention
        query = self.w_q(h_trans).reshape(
            -1,
            self.num_heads,
            self.out_ndim_trans,
        )  # Shape: (N, k, D)
        key = self.w_k(h_trans).reshape(
            -1,
            self.num_heads,
            self.out_ndim_trans,
        )  # Shape: (N, k, D)
        # value = self.w_v(H_stack)  # Shape: (N, k, D)
        value = self.w_v(h_trans).reshape(
            -1,
            self.num_heads,
            self.out_ndim_trans,
            # self.h_feats,
        )  # Shape: (N, k, D)
        # value = H.reshape(
        #     -1,
        #     1,
        #     self.h_feats,
        # )
        # query = self.w_q(self.ns[0])  # Shape: (N, k, D)
        # key = self.w_k(self.ns[0])  # Shape: (N, k, D)
        # value = self.w_v(self.ns[0])  # Shape: (N, k, D)

        # Compute attention scores (Shape: (N, k, k))
        # attention_scores = torch.matmul(
        #     query / torch.norm(query, p=2), key.t() / torch.norm(key, p=2)
        # ) / (key.size(-1) ** 0.5) + torch.eye(
        #     value.shape[0],
        #     device=value.device,
        # )

        # attention_output = self.full_attention_conv_sg(query, key, value, output_attn=False)
        # return F.relu(
        #     attention_output.reshape(
        #         attention_output.shape[0],
        #         self.out_ndim_trans * self.num_heads,
        #     )
        # ) + self.proj(h_trans)
        # attention_output = self.full_attention_conv_sg(query, key, value,h_trans, output_attn=False)
        attention_output = self.full_attention_conv_sg(query, key, value, output_attn=False)
        return attention_output



class IGNNConv(nn.Module):
    def __init__(
        self,
        nie,
        in_feats,
        h_feats,
        act,
        nas_dropout,
        layer_norm,
        N_NIE,
        transform_first,
        n_nodes,
        ndim_h_a,
        nss_dropout,
        nrl,
        n_hops,
        ndim_fc,
        no_save,
    ):
        self.no_save = no_save
        self.n_hops = n_hops
        self.h_feats = h_feats
        self.nie = nie
        self.nrl = nrl
        self.n_nodes = n_nodes
        self.device = None
        self.transform_first = transform_first
        self.adj = None
        self.adj_und = None
        self.a_feat = None

        N_NIE = self.n_hops + 1
        super().__init__()
        if nie == "deepsets":
            self.nei_ind_emb = nn.ModuleList(
                DeepSets(
                    in_feats=in_feats,
                    h_feats=h_feats,
                    acts=[acts[act]()],
                    dropout=nas_dropout,
                    layer_norm=layer_norm,
                )
                for _ in range(N_NIE)
            )
        elif nie == "gat":
            self.nei_ind_emb = nn.ModuleList(
                GATConv(
                    in_channels=in_feats,
                    out_channels=h_feats,
                    heads=1,
                    dropout=nas_dropout,
                )
                for _ in range(N_NIE)
            )

        elif nie == "gcn-nie-nst":
            if transform_first:
                self.nei_uni_emb = MLP(
                    in_feats=in_feats,
                    h_feats=[h_feats],
                    acts=[acts[act]()],
                    dropout=nas_dropout,
                    layer_norm=layer_norm,
                )
                self.nei_ind_emb = nn.ModuleList(
                    MLP(
                        in_feats=h_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nas_dropout,
                        layer_norm=layer_norm,
                    )
                    for _ in range(N_NIE)
                )
            else:
                self.nei_ind_emb = nn.ModuleList(
                    MLP(
                        in_feats=in_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nas_dropout,
                        layer_norm=layer_norm,
                    )
                    for _ in range(N_NIE)
                )
        elif nie == "gcn-nie-st":
            self.nei_ind_emb = nn.ModuleList(
                [
                    MLP(
                        in_feats=in_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nas_dropout,
                        layer_norm=layer_norm,
                    )
                ]
            )
        elif nie == "gcn-nnie-nst":
            self.nei_ind_emb = nn.ModuleList(
                (
                    GraphConv(
                        in_feats=in_feats if i == 1 else h_feats,
                        out_feats=h_feats,
                        activation=acts[act](),
                    )
                    if i != 0
                    else MLP(
                        in_feats=in_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nas_dropout,
                        layer_norm=layer_norm,
                    )
                )
                for i in range(N_NIE)
            )
        elif nie == "gcn-nnie-st":
            self.nei_ind_emb = nn.ModuleList(
                (
                    GraphConv(
                        in_feats=h_feats,
                        out_feats=h_feats,
                        activation=acts[act](),
                    )
                    if i != 0
                    else MLP(
                        in_feats=in_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nas_dropout,
                        layer_norm=layer_norm,
                    )
                )
                for i in range(N_NIE)
            )

        self.nei_feats = None

        if n_nodes is not None:
            self.w_a = MLP(
                in_feats=n_nodes,
                h_feats=[ndim_h_a],
                acts=[acts[act]()],
                dropout=0,
                layer_norm=False,
            )

        if nrl == "concat":
            self.nei_rel_learn = nn.ModuleList(
                [
                    MLP(
                        in_feats=ndim_fc,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nss_dropout,
                        layer_norm=layer_norm,
                    )
                ],
            )
        elif nrl in ["max", "sum", "mean"]:
            self.nei_rel_learn = nn.ModuleList(
                [
                    MLP(
                        in_feats=h_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nss_dropout,
                        layer_norm=layer_norm,
                    )
                ],
            )
        elif nrl == "lstm":
            self.nei_rel_learn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(nss_dropout),
                        LSTM(
                            input_dim=h_feats,
                            hidden_dim=h_feats,
                            output_dim=h_feats,
                            num_layers=1,
                            dropout=nss_dropout,
                        ),
                    )
                ],
            )
        elif nrl == "ordered-gating":
            self.linear_trans_in = ModuleList()
            self.linear_trans_out = Linear(h_feats, h_feats)
            self.norm_input = ModuleList()
            self.convs = ModuleList()

            self.tm_norm = ModuleList()
            self.tm_net = ModuleList()

            global_gating = False
            self.chunk_size = chunk_size = 64
            if global_gating == True:
                tm_net = Linear(2 * h_feats, chunk_size)

            for i in range(n_hops):
                self.tm_norm.append(LayerNorm(h_feats))
                if global_gating == True:
                    self.tm_net.append(tm_net)
                else:
                    self.tm_net.append(Linear(2 * h_feats, chunk_size))
                self.convs.append(
                    ONGNNConv(
                        tm_net=self.tm_net[i],
                        tm_norm=self.tm_norm[i],
                        simple_gating=False,
                        tm=True,
                        diff_or=True,
                        repeats=int(h_feats / chunk_size),
                    )
                )
            self.device = None

    def forward(
        self,
        device,
        graph,
        feats=None,
        batch_idx=None,
    ):
        if self.device is None:
            print("device None")
            self.to(device=device)
            self.device = device
        self.ns = []
        # adj = graph.adj()
        features = graph.ndata["feat"] if feats is None else feats
        row, col = graph.add_self_loop().edges()
        # row = row - row.min()
        if self.n_nodes is not None and self.a_feat is None:
            self.a_feat = (
                SparseTensor(
                    row=row,
                    col=col,
                    sparse_sizes=(self.n_nodes, self.n_nodes),
                )
                .to_torch_sparse_coo_tensor()
                .coalesce()
                .to(device)
            )
        if self.a_feat is not None:
            if batch_idx is not None:
                a_batch = self.a_feat.index_select(0, batch_idx.to(device))
            else:
                a_batch = self.a_feat

            h_a = self.w_a(a_batch)
        else:
            h_a = None

        if self.nie == "gcn-nnie-nst":
            h = features.to(device)
            self.ns.append(self.nei_ind_emb[0](h))
            for i in range(1, self.n_hops + 1):
                h = self.nei_ind_emb[i](graph=graph.to(device), feat=h)
                self.ns.append(h)
        elif self.nie == "gcn-nnie-st":
            h = self.nei_ind_emb[0](features.to(device))
            self.ns.append(h)
            for i in range(1, self.n_hops + 1):
                h = self.nei_ind_emb[i](graph=graph.to(device), feat=h)
                self.ns.append(h)
        elif self.nie in ["deepsets", "gcn-nie-nst", "gcn-nie-st"]:
            if self.adj_und is None:
                row, col = graph.to_simple().add_self_loop().edges()
                self.adj_und = SparseTensor(
                    row=row.long(),
                    col=col.long(),
                    sparse_sizes=(graph.num_nodes(), graph.num_nodes()),
                )
            Batch_load = False
            if (Batch_load or self.nei_feats is None) and self.no_save == False:
                # if self.no_save == False:
                self.nei_feats = preprocess_neighborhoods(
                    adj=self.adj_und,
                    features=features,
                    row_normalized=graph.row_normalized,
                    name=graph.name,
                    n_hops=self.n_hops,
                    set_diag=True,
                    remove_diag=False,
                    symm_norm={
                        "gcn": True,
                        "deepsets": False,
                        "gcn-nie-nst": True,
                        "gcn-nie-st": True,
                    }[self.nie],
                    device=torch.device("cpu"),
                    batch_idx=batch_idx if Batch_load else None,
                )
            elif self.no_save == True:
                self.nei_feats, self.adj = preprocess_neighborhoods(
                    adj=(
                        SparseTensor(
                            row=row.long(),
                            col=col.long(),
                            sparse_sizes=(graph.num_nodes(), graph.num_nodes()),
                        )
                        if self.adj is None
                        else self.adj
                    ),
                    features=features,
                    row_normalized=graph.row_normalized,
                    name=graph.name,
                    n_hops=self.n_hops,
                    set_diag=True,
                    remove_diag=False,
                    symm_norm={
                        "gcn": True,
                        "deepsets": False,
                        "gcn-nie-nst": True,
                        "gcn-nie-st": True,
                    }[self.nie],
                    device=device,
                    no_save=True,
                    return_adj=True,
                    process_adj=False if self.adj is not None else True,
                    batch_idx=batch_idx,
                )
            if self.nie in ["deepsets", "gcn-nie-nst"]:
                for i in range(self.n_hops + 1):
                    if self.transform_first:
                        self.ns.append(
                            # self.nei_ind_emb[i](self.nei_feats[i].index_select(0, batch_idx))
                            self.nei_ind_emb[i](
                                self.nei_uni_emb(
                                    self.nei_feats[i].index_select(0, batch_idx).to(self.device)
                                )
                                if batch_idx is not None
                                else self.nei_uni_emb(self.nei_feats[i].to(self.device))
                            )
                        )
                    else:
                        self.ns.append(
                            # self.nei_ind_emb[i](self.nei_feats[i].index_select(0, batch_idx))
                            self.nei_ind_emb[i](
                                self.nei_feats[i].index_select(0, batch_idx).to(self.device)
                                if batch_idx is not None
                                else self.nei_feats[i].to(self.device)
                            )
                        )
            elif self.nie in ["gcn-nie-st"]:
                for i in range(self.n_hops + 1):
                    self.ns.append(self.nei_ind_emb[0](self.nei_feats[i]))

        hops_feats = (
            torch.cat(self.ns, dim=1)
            if self.n_nodes is None
            else torch.cat((h_a, torch.cat(self.ns, dim=-1)), dim=-1)
        )

        if self.nrl == "none":
            H = self.ns[-1]

        if self.nrl == "only-concat":
            H = hops_feats

        if self.nrl == "concat":
            # return torch.cat(
            #                 [nei_rel_learn(hops_feats) for nei_rel_learn in self.nei_rel_learn],
            #                 dim=-1,
            #             )
            H = (
                torch.cat(
                    (
                        torch.cat(
                            [nei_rel_learn(hops_feats) for nei_rel_learn in self.nei_rel_learn],
                            dim=-1,
                        ),
                        h_a,
                    ),
                    dim=-1,
                )
                if self.n_nodes is not None
                else torch.cat(
                    [nei_rel_learn(hops_feats) for nei_rel_learn in self.nei_rel_learn],
                    dim=-1,
                )
            )
        if self.nrl == "max":
            H = self.nei_rel_learn[0](
                torch.max(
                    torch.stack(self.ns, dim=1),
                    dim=1,
                )[0]
            )
        if self.nrl == "mean":
            H = self.nei_rel_learn[0](
                torch.mean(
                    torch.stack(self.ns, dim=1),
                    dim=1,
                )
            )
        if self.nrl == "sum":
            H = self.nei_rel_learn[0](
                torch.sum(
                    torch.stack(self.ns, dim=1),
                    dim=1,
                )
            )
        if self.nrl == "lstm":
            H = self.nei_rel_learn[0](
                torch.stack(self.ns, dim=1),
            )
        if self.nrl == "ordered-gating":
            check_signal = []
            h = self.ns[0]
            tm_signal = h.new_zeros(self.chunk_size)
            for j, conv in enumerate(self.convs):
                h = F.dropout(h, p=0.1, training=self.training)
                m = F.dropout(self.ns[j + 1], p=0.1, training=self.training)
                h, tm_signal = conv(
                    h,
                    m,
                    last_tm_signal=tm_signal,
                )
                check_signal.append(dict(zip(["tm_signal"], [tm_signal])))
            H = h

        return H, hops_feats, h_a


class FlatGNN(nn.Module):
    """FlatGNN."""

    def __init__(
        self,
        in_feats,
        h_feats,
        n_hops,
        nas_dropout=0.0,
        nss_dropout=0.8,
        n_intervals=3,
        out_ndim_trans=64,
        no_save=False,
        nie="gcn-nie-nst",
        nrl="concat",
        act="relu",
        layer_norm=True,
        n_nodes=None,
        ndim_h_a=64,
        num_heads=1,
        transform_first=False,
        trans_layer_num=3,
        ignn_layer_num=1,
    ):
        super().__init__()
        self.no_save = no_save
        self.n_hops = n_hops
        self.h_feats = h_feats
        self.nie = nie
        self.nrl = nrl
        self.n_nodes = n_nodes
        self.device = None
        self.transform_first = transform_first
        self.num_heads = num_heads
        self.out_ndim_trans = out_ndim_trans

        N_NIE = self.n_hops + 1

        if n_nodes is not None:
            ndim_fc = h_feats * N_NIE + ndim_h_a
            # ndim_fc = h_feats * N_NIE
        else:
            ndim_fc = h_feats * N_NIE

        # elif nrl == "self-attention":

        nrl = "concat"

        self.ignnconvs = nn.ModuleList(
            [
                IGNNConv(
                    nie,
                    in_feats if i == 0 else h_feats,
                    h_feats,
                    act,
                    nas_dropout,
                    layer_norm,
                    N_NIE,
                    transform_first,
                    n_nodes,
                    ndim_h_a,
                    nss_dropout,
                    nrl=nrl,
                    n_hops=n_hops,
                    ndim_fc=ndim_fc,
                    no_save=i != 0,
                )
                for i in range(ignn_layer_num)
            ]
        )
        trans_layer_num = trans_layer_num
        # in_ndim_trans = ndim_fc

        if n_nodes is not None:
            self.in_ndim_trans = {"concat": h_feats + ndim_h_a, "only-concat": ndim_fc}[nrl]
            # ndim_fc = h_feats * N_NIE
        else:
            self.in_ndim_trans = {"concat": h_feats, "only-concat": ndim_fc}[nrl]

        self.igformers = nn.ModuleList(
            [
                IGFormer(
                    in_ndim_trans=(
                        self.in_ndim_trans if i == 0 else self.out_ndim_trans * self.num_heads
                    ),
                    out_ndim_trans=out_ndim_trans,
                    num_heads=num_heads,
                    trans_dropout=0.2,
                )
                for i in range(trans_layer_num)
            ]
        )
        self.fc = MLP(
            in_feats=self.in_ndim_trans,
            h_feats=[self.out_ndim_trans * self.num_heads],
            acts=[acts[act]()],
            dropout=nss_dropout,
            layer_norm=layer_norm,
        )

    def forward(
        self,
        graph: dgl.DGLGraph,
        device: torch.device,
        feats=None,
        batch_idx=None,
    ):
        H = None
        for i, ignnconvs in enumerate(self.ignnconvs):
            H, hops_feats, h_a = ignnconvs(
                device=device,
                graph=graph,
                feats=feats if i == 0 else H,
                batch_idx=batch_idx,
            )

        # h_trans = hops_feats
        h_trans = H
        # H = self.fc(H_stack)
        # h_trans = H_stack if self.in_ndim_trans == H_stack.shape[1] else H
        # H_d = H.detach()

        for idx, igformer in enumerate(self.igformers):
            attention_output = igformer(h_trans)
            h_trans = attention_output + self.fc(h_trans) if idx==0 else attention_output + h_trans
            # h_trans = attention_output.mean(dim=-1)
            # print(attention_output.shape, h_trans.shape)

        return h_trans
        # return self.fc(torch.cat((H, h_trans), dim=-1))
        # return self.proj(torch.cat((H, h_a, attention_output.mean(dim=1)), dim=-1))
        # return 0.5*H + 0.5*attention_output.mean(dim=1)


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.t())
    return sim_mt

    # # compute attention for visualization if needed
    # if output_attn:
    #     attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
    #     normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
    #     attention = attention / normalizer

    # if output_attn:
    #     return attn_output, attention
    # else:
    #     return attn_output


# def full_attention_conv(qs, ks, vs, output_attn=False):
#     # normalize input
#     qs = qs / torch.norm(qs, p=2)  # [N, H, M]
#     ks = ks / torch.norm(ks, p=2)  # [L, H, M]
#     N = qs.shape[0]

#     # numerator
#     kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
#     attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
#     attention_num += N * vs

#     # denominator
#     all_ones = torch.ones([ks.shape[0]]).to(ks.device)
#     ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
#     attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

#     # attentive aggregated results
#     attention_normalizer = torch.unsqueeze(
#         attention_normalizer, len(attention_normalizer.shape)
#     )  # [N, H, 1]
#     attention_normalizer += torch.ones_like(attention_normalizer) * N
#     attn_output = attention_num / attention_normalizer  # [N, H, D]

#     # compute attention for visualization if needed
#     if output_attn:
#         attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
#         normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
#         attention = attention / normalizer

#     if output_attn:
#         return attn_output, attention
#     else:
#         return attn_output
