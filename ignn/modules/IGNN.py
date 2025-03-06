"""FaltGNN Layer."""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals,invalid-name,too-many-branches,too-many-statements,
import os
from pathlib import Path
from typing import Generator, Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d, LayerNorm, Linear, ModuleList
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GATConv, MessagePassing, SAGEConv
from torch_sparse import remove_diag, set_diag
from torch_sparse.tensor import SparseTensor

from .Conf import INConf
from .GCNConv import GCNConv
from .LSTM import LSTM
from .OrderedGating import ONGNNConv

# from dgl.nn.pytorch import GraphConv
# from torch_geometric.data import Data
# from torch_geometric.utils import to_dgl

acts = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "sigmoid": nn.Sigmoid,
    "none": nn.Identity,
    "tanh": nn.Tanh,
}


def preprocess_adj(adj, add_self_loop, remove_self_loop, symm_norm):
    if add_self_loop:
        adj = set_diag(adj)
    if remove_self_loop:
        adj = remove_diag(adj)
    if symm_norm:
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    else:
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float("inf")] = 0
        adj = deg_inv.view(-1, 1) * adj
    return adj


class IGNNConv(nn.Module):
    """IGNN Conv."""

    def __init__(
        self,
        IN,
        in_feats,
        h_feats,
        act,
        nas_dropout,
        transform_first,
        nss_dropout,
        RN,
        n_hops,
        ndim_fc,
        norm: Optional[Literal["bn", "ln"]] = None,
    ):
        super().__init__()
        self.n_hops = n_hops
        self.h_feats = h_feats
        self.IN = IN
        self.RN = RN
        self.transform_first = transform_first
        self.n_nie = self.n_hops + 1
        act_func = acts[act]()
        self.nei_feats = None
        self.adj = None
        self.device = None
        self.norm = lambda x: (
            {"ln": LayerNorm, "bn": BatchNorm1d}[norm](x) if norm else nn.Identity()
        )

        if IN == "gcn":
            self.nei_ind_emb = nn.ModuleList(
                (
                    GCNConv(
                        in_channels=h_feats,
                        out_channels=h_feats,
                        improved=False,
                        cached=False,
                        add_self_loops=False,
                        normalize=True,
                        weight=not self.RN == "attentive",
                        # weight=True,
                        bias=True,
                        act=act_func,
                    )
                    # GraphConv(
                    #     in_feats=h_feats,
                    #     out_feats=h_feats,
                    #     norm='both',
                    #     weight=False if RN == "attentive" else True,
                    #     activation=act_func,
                    #     allow_zero_in_degree=RN=="attentive",
                    # )
                    if i != 0
                    else nn.Sequential(
                        nn.Dropout(p=nas_dropout),
                        nn.Linear(in_feats, h_feats),
                        self.norm(h_feats),
                        act_func,
                    )
                )
                for i in range(self.n_nie)
            )
        elif IN == "gcn-IN-SN":
            if transform_first:
                self.nei_uni_emb = nn.Sequential(
                    nn.Dropout(p=nas_dropout),
                    nn.Linear(in_feats, h_feats),
                    self.norm(h_feats),
                    act_func,
                )
            self.nei_ind_emb = nn.ModuleList(
                nn.Sequential(
                    nn.Dropout(p=nas_dropout),
                    nn.Linear(h_feats if transform_first else in_feats, h_feats),
                    self.norm(h_feats),
                    act_func,
                )
                for _ in range(self.n_nie)
            )
        else:
            self.init_custom_INs(IN, in_feats, h_feats, act_func, nas_dropout)

        if RN == "concat":
            self.nei_rel_learn = nn.Sequential(
                nn.Dropout(p=nss_dropout),
                nn.Linear(ndim_fc, h_feats),
                self.norm(h_feats),
                act_func,
            )
        elif RN == "residual":
            # no nei_rel_learn needed for residual variant
            pass
        elif RN == "attentive":
            self.nei_rel_learn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(p=nss_dropout),
                        nn.Linear(h_feats * 2, 1),
                        self.norm(1),
                        acts["tanh"](),
                    )
                    for _ in range(self.n_nie)
                ]
            )
        else:
            self.init_custom_RNs(RN, h_feats, act_func, nss_dropout, n_hops)

        self.reset_parameters()

    def init_custom_INs(self, IN, in_feats, h_feats, act_func, nas_dropout):
        if IN == "gcn-IN-nSN":
            self.nei_ind_emb = nn.Sequential(
                nn.Dropout(p=nas_dropout),
                nn.Linear(in_feats, h_feats),
                self.norm(h_feats),
                act_func,
            )
        elif IN == "gcn-nIN-SN":
            self.nei_ind_emb = nn.ModuleList(
                (
                    GCNConv(
                        in_channels=in_feats if i == 1 else h_feats,
                        out_channels=h_feats,
                        improved=False,
                        cached=False,
                        add_self_loops=True,
                        normalize=True,
                        weight=True,
                        bias=True,
                        act=act_func,
                    )
                    if i != 0
                    else nn.Sequential(
                        nn.Dropout(p=nas_dropout),
                        nn.Linear(in_feats, h_feats),
                        self.norm(h_feats),
                        act_func,
                    )
                )
                for i in range(self.n_nie)
            )
        elif IN == "gcn-nIN-nSN":
            self.nei_ind_emb = nn.ModuleList(
                (
                    GCNConv(
                        in_channels=in_feats if i == 1 else h_feats,
                        out_channels=h_feats,
                        improved=False,
                        cached=False,
                        add_self_loops=True,
                        normalize=True,
                        weight=True,
                        bias=True,
                        act=act_func,
                    )
                    if i != 0
                    else nn.Sequential(
                        nn.Dropout(p=nas_dropout),
                        nn.Linear(in_feats, h_feats),
                        self.norm(h_feats),
                        act_func,
                    )
                )
                for i in range(self.n_nie)
            )
        else:
            raise ValueError(f"IN={IN} is not supported.")

    def init_custom_RNs(self, RN, h_feats, act_func, nss_dropout, n_hops):
        if RN in ["max", "sum", "mean"]:
            self.nei_rel_learn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(p=nss_dropout),
                        nn.Linear(h_feats, h_feats),
                        self.norm(h_feats),
                        act_func,
                    )
                ],
            )
        elif RN == "lstm":
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
        elif RN == "ordered-gating":
            self.linear_trans_in = ModuleList()
            self.linear_trans_out = Linear(h_feats, h_feats)
            self.norm_input = ModuleList()
            self.convs = ModuleList()

            self.tm_norm = ModuleList()
            self.tm_net = ModuleList()

            global_gating = False
            self.chunk_size = chunk_size = 64
            tm_net = Linear(2 * h_feats, chunk_size) if global_gating else None

            for i in range(n_hops):
                self.tm_norm.append(LayerNorm(h_feats))
                self.tm_net.append(tm_net if global_gating else Linear(2 * h_feats, chunk_size))
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
        elif RN == "none":
            pass
        else:
            raise ValueError(f"RN={RN} is not supported.")

    def get_leaf_modules(self, module: nn.Module) -> Generator[nn.Module, nn.Module, nn.Module]:
        if len(list(module.children())) == 0:
            yield module
        for m in module.children():
            yield from self.get_leaf_modules(module=m)

    def reset_parameters(self):
        for module in self.get_leaf_modules(module=self):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    @staticmethod
    def inceptive_aggregation(
        adj: SparseTensor,
        features: torch.FloatTensor,
        IN_config: INConf,
        preprocess: bool,
        device: torch.device = torch.device("cpu"),
    ):
        (
            name,
            n_hops,
            add_self_loop,
            remove_self_loop,
            symm_norm,
            row_normalized,
            fast,
            save_dir,
        ) = vars(IN_config).values()
        adj = preprocess_adj(adj, add_self_loop, remove_self_loop, symm_norm) if preprocess else adj
        adj = adj.to_torch_sparse_csr_tensor() if isinstance(adj, SparseTensor) else adj
        assert isinstance(adj, torch.Tensor), "adj should be torch.tensor"
        features = (F.normalize(features, dim=1) if row_normalized else features).to(device)
        adj = adj.to(device)

        nei_feats = [features]
        x = features

        # c-IGNN
        if not fast:
            for i in range(1, n_hops + 1):
                x = torch.spmm(adj, x).to(torch.float)
                nei_feats.append(x)
            return nei_feats, adj

        # Fast c-IGNN: caching aggregated features for faster training
        sym = "sym" if symm_norm else "nsy"
        diag = "diag" if add_self_loop else "ndiag"
        diag = "ndiag" if remove_self_loop else diag
        norm = "norm" if row_normalized else "nnorm"
        base = Path(f"{save_dir}/{name}/{norm}/{diag}/{sym}")
        base.mkdir(parents=True, exist_ok=True)

        hops = os.listdir(base) if base.exists() else []
        if f"{n_hops}" not in hops:
            for i in range(1, n_hops + 1):
                file = base.joinpath(f"{i}")
                if file.exists():
                    x = torch.load(file, map_location=device)
                else:
                    x = torch.spmm(adj, x)
                    torch.save(obj=x.to(torch.float), f=file, pickle_protocol=4)

        print(f"Load aggregated feats from {base}")
        for i in range(1, n_hops + 1):
            nei_feats.append(torch.load(base.joinpath(f"{i}"), map_location=device))

        return nei_feats, adj

    def custom_INs(self, features, edge_index, device):
        if self.IN == "gcn-nIN-SN":
            h = features
            ns = [self.nei_ind_emb[0](h)]
            for i in range(1, self.n_nie):
                h = self.nei_ind_emb[i](x=h, edge_index=edge_index)
                ns.append(h)
        elif self.IN == "gcn-nIN-nSN":
            h = self.nei_ind_emb[0](features)
            ns = [h]
            for i in range(1, self.n_nie):
                h = self.nei_ind_emb[i](x=h, edge_index=edge_index)
                ns.append(h)
        return ns

    def custom_RNs(self, ns):
        if self.RN == "max":
            return self.nei_rel_learn[0](torch.max(torch.stack(ns, dim=1), dim=1)[0])
        if self.RN == "mean":
            return self.nei_rel_learn[0](torch.mean(torch.stack(ns, dim=1), dim=1))
        if self.RN == "sum":
            return self.nei_rel_learn[0](torch.sum(torch.stack(ns, dim=1), dim=1))
        if self.RN == "lstm":
            return self.nei_rel_learn[0](torch.stack(ns, dim=1))
        if self.RN == "ordered-gating":
            check_signal = []
            h = ns[0]
            tm_signal = h.new_zeros(self.chunk_size)
            for j, conv in enumerate(self.convs):
                h = F.dropout(h, p=0.1, training=self.training)
                m = F.dropout(ns[j + 1], p=0.1, training=self.training)
                h, tm_signal = conv(h, m, last_tm_signal=tm_signal)
                check_signal.append(dict(zip(["tm_signal"], [tm_signal])))
            return h

        return torch.cat(ns, dim=1)

    def forward(self, edge_index, features, IN_config: INConf, device=torch.device("cpu")):
        self.to(device=device)
        self.device = device

        if not isinstance(edge_index, SparseTensor):
            n_nodes = features.shape[0]
            edge_index = SparseTensor(
                row=edge_index[0].long(), col=edge_index[1].long(), sparse_sizes=(n_nodes, n_nodes)
            )

        if self.IN == "gcn":
            if self.RN == "residual":
                h = self.nei_ind_emb[0](features)
                for i in range(1, self.n_nie):
                    h = self.nei_ind_emb[i](x=h, edge_index=edge_index) + h
                return h

            if self.RN == "attentive":
                h = self.nei_ind_emb[0](features)
                for i in range(1, self.n_nie):
                    h_k = self.nei_ind_emb[i](x=h, edge_index=edge_index)
                    # h_k = self.nei_ind_emb[i](graph=to_dgl(Data(edge_index=edge_index, num_nodes=features.shape[0])), feat=h)
                    a = self.nei_rel_learn[i](torch.cat([h_k, h], dim=-1))
                    h = a * h_k + (1 - a) * h
                return h

            if self.RN == "none":
                h = self.nei_ind_emb[0](features)
                for i in range(1, self.n_nie):
                    h = self.nei_ind_emb[i](x=h, edge_index=edge_index)
                return h

        if not IN_config.fast or self.nei_feats is None:
            self.nei_feats, _ = self.inceptive_aggregation(
                adj=edge_index,
                features=features,
                IN_config=IN_config,
                preprocess=True,
                device=device,
            )

        if self.IN == "gcn-IN-SN":
            if self.transform_first:
                ns = [
                    self.nei_ind_emb[i](self.nei_uni_emb(self.nei_feats[i]))
                    for i in range(self.n_nie)
                ]
            else:
                ns = [self.nei_ind_emb[i](self.nei_feats[i]) for i in range(self.n_nie)]
        elif self.IN == "gcn-IN-nSN":
            ns = [self.nei_ind_emb(self.nei_feats[i]) for i in range(self.n_nie)]
        else:
            ns = self.custom_INs(features=features, edge_index=edge_index, device=device)

        if self.RN == "concat":
            ns = torch.cat(ns, dim=1)
            return self.nei_rel_learn(ns)
        return self.custom_RNs(ns)
