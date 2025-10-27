"""IGNNConv Layer."""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals,invalid-name,too-many-branches,too-many-statements,
import os
from pathlib import Path
from typing import Dict, Type

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv
from torch_sparse import remove_diag, set_diag
from torch_sparse.tensor import SparseTensor

from ..utils import get_logger
from .GCNConv import GCNConv
from .GCNIncep import GCNIncep

logger = get_logger(__name__)

ACT_DICT = {
    "relu": nn.ReLU(),
    "prelu": nn.PReLU(),
    "leakyrelu": nn.LeakyReLU(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=-1),
    "tanh": nn.Tanh(),
    "none": nn.Identity(),
}

NORM_DICT: Dict[str, Type[nn.Module]] = {
    "ln": nn.LayerNorm,
    "bn": nn.BatchNorm1d,
    "none": nn.Identity,
}

CONV_DICT: Dict[str, Type[nn.Module]] = {
    "gcn": GCNConv,
    "sage": SAGEConv,
    "gat": GATConv,
    # NOTE: Inceptive GCN AGG() without transformation matrices Ws
    "gcn_incep": GCNIncep,
}


class GNNConv(torch.nn.Module):
    """GNN Convs."""

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout: float = 0,
        norm_type: str = "none",
        act_type: str = "relu",
        agg_type: str = "gcn",
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalization = NORM_DICT[norm_type](in_channels)
        self.activation = ACT_DICT[act_type]
        self.conv = CONV_DICT[agg_type](
            in_channels, out_channels, normalization=self.normalization, act=self.activation
        )

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return self.dropout(x)


class IGNNConv(nn.Module):
    """IGNN Conv."""

    def __init__(
        self,
        IN,
        in_feats,
        h_feats,
        pre_dropout,
        hid_dropout,
        RN,
        n_hops,
        norm_type: str = "none",
        act_type: str = "relu",
        agg_type: str = "gcn",
        att_act_type: str = "tanh",
        fast: bool = False,
        pre_lin: bool = False,
    ):
        super().__init__()
        self.n_hops = n_hops
        self.h_feats = h_feats
        self.IN = IN
        self.RN = RN
        self.act_type = act_type
        self.pre_lin = pre_lin
        self.norm_type = norm_type
        self.act_type = act_type
        self.agg_type = agg_type
        self.nei_feats = None
        self.adj = None
        self.device = None
        self.lin = None

        self.init_INs(IN, in_feats, h_feats, pre_dropout, hid_dropout, fast)
        self.init_RNs(RN, h_feats, hid_dropout, h_feats * (n_hops + 1), att_act_type)
        self.reset_parameters()

    def get_leaf_modules(self, module):
        if len(list(module.children())) == 0:
            yield module
        for m in module.children():
            yield from self.get_leaf_modules(module=m)

    def reset_parameters(self):
        for module in self.get_leaf_modules(module=self):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def init_INs(self, IN, in_feats, h_feats, pre_dropout, hid_dropout, fast):
        if IN == "nIN-nSN":
            assert (
                self.RN == "none" and self.agg_type == "gcn"
            ), f"{IN} only supports RN==none and agg_type==gcn."
            self.inceptive_agg = nn.ModuleList()
            for i in range(self.n_hops + 1):
                self.inceptive_agg.append(
                    GNNConv(h_feats, h_feats, 0, "none", self.act_type, "gcn")
                    if i != 0
                    else nn.Sequential(
                        nn.Dropout(p=pre_dropout),
                        nn.Linear(in_feats, h_feats),
                        NORM_DICT[self.norm_type](h_feats),
                        ACT_DICT[self.act_type],
                    )
                )
        elif IN == "IN-nSN":
            if self.RN == "none":
                self.lin = nn.Sequential(
                    nn.Dropout(p=pre_dropout),
                    nn.Linear(in_feats, h_feats),
                    NORM_DICT[self.norm_type](h_feats),
                    ACT_DICT[self.act_type],
                )
                self.inceptive_agg = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Dropout(p=hid_dropout),
                            nn.Linear(h_feats, h_feats),
                            NORM_DICT[self.norm_type](h_feats),
                            ACT_DICT[self.act_type],
                        )
                    ]
                )
                for _ in range(self.n_hops):
                    self.inceptive_agg.append(
                        nn.ModuleList(
                            [
                                GNNConv(h_feats, h_feats, 0, "none", "none", self.agg_type),
                                nn.Sequential(
                                    nn.Dropout(hid_dropout),
                                    nn.Linear(h_feats, h_feats),
                                    NORM_DICT[self.norm_type](h_feats),
                                    ACT_DICT[self.act_type],
                                ),
                            ]
                        )
                    )
            else:
                self.inceptive_agg = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Dropout(p=pre_dropout),
                            nn.Linear(in_feats, h_feats),
                            NORM_DICT[self.norm_type](h_feats),
                            ACT_DICT[self.act_type],
                        )
                    ]
                )
                for _ in range(self.n_hops):
                    # NOTE: Since our theoretical analysis omits LN/BN normalization and activation\
                    # in the AGG hidden layers while allowing them in the final output layer, we follow\
                    # this practice in the implementation of the three variants. Reintroducing normalization\
                    # into the AGG hidden layers may further improve performance, as some studies have shown\
                    # its role in mitigating oversmoothing. However, this restriction may also limit the model's\
                    # ability to adaptively control smoothness, which we leave for future work.
                    self.inceptive_agg.append(
                        GNNConv(h_feats, h_feats, 0, "none", "none", self.agg_type)
                    )
        elif IN == "IN-SN":
            if self.pre_lin and not fast:
                self.lin = nn.Sequential(
                    nn.Dropout(p=pre_dropout),
                    nn.Linear(in_feats, h_feats),
                    NORM_DICT[self.norm_type](h_feats),
                    ACT_DICT[self.act_type],
                )
            n_feats = in_feats if not self.lin else h_feats
            self.inceptive_agg = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(p=pre_dropout if not self.lin else hid_dropout),
                        nn.Linear(n_feats, h_feats),
                        NORM_DICT[self.norm_type](h_feats),
                        ACT_DICT[self.act_type],
                    )
                ]
            )
            for _ in range(self.n_hops):
                self.inceptive_agg.append(
                    nn.Sequential(
                        nn.Dropout(p=pre_dropout if not self.lin else hid_dropout),
                        nn.Linear(n_feats, h_feats),
                        NORM_DICT[self.norm_type](h_feats),
                        ACT_DICT[self.act_type],
                    )
                    if fast
                    else nn.ModuleList(
                        # NOTE: Since our theoretical analysis omits LN/BN normalization and activation\
                        # in the AGG hidden layers while allowing them in the final output layer, we follow\
                        # this practice in the implementation of the three variants. Reintroducing normalization\
                        # into the AGG hidden layers may further improve performance, as some studies have shown\
                        # its role in mitigating oversmoothing. However, this restriction may also limit the model's\
                        # ability to adaptively control smoothness, which we leave for future work.
                        [
                            GNNConv(n_feats, n_feats, 0, "none", "none", self.agg_type),
                            nn.Sequential(
                                nn.Dropout(p=pre_dropout if not self.lin else hid_dropout),
                                nn.Linear(n_feats, h_feats),
                                NORM_DICT[self.norm_type](h_feats),
                                ACT_DICT[self.act_type],
                            ),
                        ]
                    )
                )

    def init_RNs(self, RN, h_feats, hid_dropout, ndim_fc, att_act_type):
        if RN == "concat":
            self.nei_rel_learn = nn.Sequential(
                nn.Dropout(p=hid_dropout),
                nn.Linear(ndim_fc, h_feats),
                NORM_DICT[self.norm_type](h_feats),
                ACT_DICT[self.act_type],
            )
        elif RN in ["residual", "none"]:
            # no nei_rel_learn needed for residual variant
            self.nei_rel_learn = nn.Identity()
        elif RN == "attentive":
            self.nei_rel_learn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(p=hid_dropout),
                        nn.Linear(h_feats * 2, 1),
                        NORM_DICT[self.norm_type](1),
                        ACT_DICT[att_act_type],
                    )
                    for _ in range(self.n_hops + 1)
                ]
            )

    # pylint: disable=too-many-return-statements
    def forward(self, edge_index, features, IN_config, device=torch.device("cpu"), hiddens=False):
        self.to(device=device)
        self.device = device

        if not isinstance(edge_index, SparseTensor):
            n_nodes = features.shape[0]
            edge_index = SparseTensor(
                row=edge_index[0].long(), col=edge_index[1].long(), sparse_sizes=(n_nodes, n_nodes)
            )

        if self.IN == "nIN-nSN":
            # NOTE: GCN - ❌ IN ❌ SN ❌ RN
            if self.RN == "none":
                h = self.inceptive_agg[0](features)
                if hiddens:
                    hs = [h.detach().clone()]
                for i in range(1, self.n_hops + 1):
                    h = self.inceptive_agg[i](x=h, edge_index=edge_index)
                    if hiddens:
                        hs.append(h.detach().clone())
                return (h, hs) if hiddens else h

        if self.IN == "IN-nSN":
            # NOTE: r-IGNN - ✅ IN ❌ SN ✅ RN
            if self.RN == "residual":
                h = self.inceptive_agg[0](features)
                if hiddens:
                    hs = [h.detach().clone()]
                for i in range(1, self.n_hops + 1):
                    h = self.inceptive_agg[i](x=h, edge_index=edge_index) + h
                    if hiddens:
                        hs.append(h.detach().clone())
                return (h, hs) if hiddens else h

            # NOTE: a-IGNN - ✅ IN ❌ SN ✅ RN
            if self.RN == "attentive":
                h = self.inceptive_agg[0](features)
                if hiddens:
                    hs = [h.detach().clone()]
                for i in range(1, self.n_hops + 1):
                    h_k = self.inceptive_agg[i](x=h, edge_index=edge_index)
                    a = self.nei_rel_learn[i](torch.cat([h_k, h], dim=-1))
                    h = a * h_k + (1 - a) * h
                    if hiddens:
                        hs.append(h.detach().clone())
                return (h, hs) if hiddens else h

            # NOTE: SIGN w/o SN - ✅ IN ❌ SN ❌ RN
            if self.RN == "none":
                h = self.lin(features)
                ns = [self.inceptive_agg[0](h)]
                _h = h
                for i in range(1, self.n_hops + 1):
                    _h = self.inceptive_agg[i][0](x=_h, edge_index=edge_index)
                    ns.append(self.inceptive_agg[i][1](_h))
                return torch.cat(ns, dim=1)

        if self.IN == "IN-SN":
            if IN_config.fast:
                self.nei_feats, _ = (
                    fast_inceptive_aggregation(
                        adj=edge_index,
                        features=features,
                        IN_config=IN_config,
                        preprocess=True,
                        device=device,
                    )
                    if self.nei_feats is None
                    else (self.nei_feats, None)
                )
                ns = [self.inceptive_agg[i](self.nei_feats[i]) for i in range(self.n_hops + 1)]
            else:
                ns = [self.inceptive_agg[0](self.lin(features) if self.lin else features)]
                _h = self.lin(features) if self.lin else features
                for i in range(1, self.n_hops + 1):
                    _h = self.inceptive_agg[i][0](x=_h, edge_index=edge_index)
                    ns.append(self.inceptive_agg[i][1](_h))

            return {
                # NOTE: c-IGNN - ✅ IN ✅ SN ✅ RN
                "concat": (
                    (self.nei_rel_learn(torch.cat(ns, dim=1)), ns)
                    if hiddens
                    else self.nei_rel_learn(torch.cat(ns, dim=1))
                ),
                # NOTE: SIGN - ✅ IN ✅ SN 〰 RN (merged with SN)
                "none": torch.cat(ns, dim=1),
            }[self.RN]

        raise ValueError(f"Either IN: {self.IN} or RN: {self.RN} is not valid.")

    def get_Ws(self):
        if self.RN in ["residual", "none"]:
            Ws = [self.inceptive_agg[0][1].weight.detach().clone()]
            for i in range(1, len(self.inceptive_agg)):
                Ws.append(self.inceptive_agg[i].conv.lin.weight.detach().clone())
            return Ws

        if self.RN == "concat":
            Ws = []
            for i, inceptive_agg in enumerate(self.inceptive_agg):
                Ws.append(inceptive_agg[1].weight.detach().clone())
            return Ws, self.nei_rel_learn[1].weight.detach().clone()

        raise ValueError(f'{self.RN} not in ["none", "residual", "attentive", "concat"]')


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


def fast_inceptive_aggregation(adj, features, IN_config, preprocess, device=torch.device("cpu")):
    (
        name,
        n_hops,
        add_self_loop,
        remove_self_loop,
        symm_norm,
        row_normalized,
        _,
        save_dir,
    ) = vars(IN_config).values()
    logger.info("running fast c-IGNN with caching")

    adj = preprocess_adj(adj, add_self_loop, remove_self_loop, symm_norm) if preprocess else adj
    adj = adj.to_torch_sparse_csr_tensor() if isinstance(adj, SparseTensor) else adj
    adj = adj.to(device)
    assert isinstance(adj, torch.Tensor), "adj should be torch.tensor"
    features = (F.normalize(features, dim=1) if row_normalized else features).to(device)
    nei_feats = [features]
    x = features

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
            if not file.exists():
                x = torch.load(base.joinpath(f"{i-1}"), map_location=device) if i > 1 else x
                x = torch.spmm(adj, x)
                torch.save(obj=x.to(torch.float), f=file, pickle_protocol=4)

    logger.info("%s", f"Load aggregated feats from {base}")
    for i in range(1, n_hops + 1):
        nei_feats.append(torch.load(base.joinpath(f"{i}"), map_location=device))

    return nei_feats, adj
