"""IGNNConv Layer."""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals,invalid-name,too-many-branches,too-many-statements,
import os
from pathlib import Path
from typing import Callable, Generator, Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d, LayerNorm
from torch_sparse import remove_diag, set_diag
from torch_sparse.tensor import SparseTensor

from ..configs import INConf
from ..utils import get_logger
from .GCNConv import GCNConv

logger = get_logger(__name__)

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

    logger.info("running fast c-IGNN with caching")

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

    logger.info("%s", f"Load aggregated feats from {base}")
    for i in range(1, n_hops + 1):
        nei_feats.append(torch.load(base.joinpath(f"{i}"), map_location=device))

    return nei_feats, adj


class IGNNConv(nn.Module):
    """IGNN Conv."""

    def __init__(
        self,
        IN,
        in_feats,
        h_feats,
        act,
        nas_dropout,
        nss_dropout,
        RN,
        n_hops,
        ndim_fc,
        act_att="tanh",
        norm: Optional[Literal["bn", "ln"]] = None,
    ):
        super().__init__()
        self.n_hops = n_hops
        self.h_feats = h_feats
        self.IN = IN
        self.RN = RN
        self.n_nie = self.n_hops + 1
        act_func = acts[act]()
        self.nei_feats = None
        self.adj = None
        self.device = None
        self.norm = lambda x: (
            {"ln": LayerNorm, "bn": BatchNorm1d}[norm](x) if norm else nn.Identity()
        )

        self.init_INs(
            IN=IN,
            in_feats=in_feats,
            h_feats=h_feats,
            act_func=act_func,
            nas_dropout=nas_dropout,
        )

        self.init_RNs(
            RN=RN,
            h_feats=h_feats,
            act_func=act_func,
            nss_dropout=nss_dropout,
            ndim_fc=ndim_fc,
            act_att=act_att,
        )

        self.reset_parameters()

    def get_leaf_modules(self, module: nn.Module) -> Generator[nn.Module, nn.Module, nn.Module]:
        if len(list(module.children())) == 0:
            yield module
        for m in module.children():
            yield from self.get_leaf_modules(module=m)

    def reset_parameters(self):
        for module in self.get_leaf_modules(module=self):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def init_INs(
        self,
        IN: str,
        in_feats: int,
        h_feats: int,
        act_func: nn.Module,
        nas_dropout: float,
    ):
        """Initialize INs. no SN is possible without IN.

        Args:
            IN (str): IN type.
            in_feats (int): in_feat dims.
            h_feats (int): h_feats dims.
            act_func (Callable): act func.
            nas_dropout (float): dropout.
        """
        if IN == "gcn-nIN-nSN":
            self.nei_ind_emb = nn.ModuleList(
                (
                    GCNConv(
                        in_channels=h_feats,
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
        elif IN == "gcn-IN-nSN":
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
        elif IN == "gcn-IN-SN":
            self.nei_ind_emb = nn.ModuleList(
                nn.Sequential(
                    nn.Dropout(p=nas_dropout),
                    nn.Linear(in_feats, h_feats),
                    self.norm(h_feats),
                    act_func,
                )
                for _ in range(self.n_nie)
            )

    def init_RNs(self, RN, h_feats, act_func, nss_dropout, ndim_fc, act_att):
        if RN == "concat":
            self.nei_rel_learn = nn.Sequential(
                nn.Dropout(p=nss_dropout),
                nn.Linear(ndim_fc, h_feats),
                self.norm(h_feats),
                act_func,
            )
        elif RN in ["residual", "none"]:
            # no nei_rel_learn needed for residual variant
            pass
        elif RN == "attentive":
            self.nei_rel_learn = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(p=nss_dropout),
                        nn.Linear(h_feats * 2, 1),
                        self.norm(1),
                        acts[act_att](),
                    )
                    for _ in range(self.n_nie)
                ]
            )

    # pylint: disable=too-many-return-statements
    def forward(
        self, edge_index, features, IN_config: INConf, device=torch.device("cpu"), hiddens=False
    ):
        self.to(device=device)
        self.device = device

        if not isinstance(edge_index, SparseTensor):
            n_nodes = features.shape[0]
            edge_index = SparseTensor(
                row=edge_index[0].long(), col=edge_index[1].long(), sparse_sizes=(n_nodes, n_nodes)
            )

        if self.IN == "gcn-nIN-nSN":
            h = self.nei_ind_emb[0](features)
            hs = [h.detach().clone()] if hiddens else None
            for i in range(1, self.n_nie):
                h = self.nei_ind_emb[i](x=h, edge_index=edge_index)
                if hiddens:
                    hs.append(h.detach().clone())

            return (h, hs) if hiddens else h

        if self.IN == "gcn-IN-nSN":
            if self.RN == "residual":
                h = self.nei_ind_emb[0](features)
                if hiddens:
                    hs = [h.detach().clone()]
                for i in range(1, self.n_nie):
                    h = self.nei_ind_emb[i](x=h, edge_index=edge_index) + h
                    if hiddens:
                        hs.append(h.detach().clone())
                return (h, hs) if hiddens else h

            if self.RN == "attentive":
                h = self.nei_ind_emb[0](features)
                if hiddens:
                    hs = [h.detach().clone()]
                for i in range(1, self.n_nie):
                    h_k = self.nei_ind_emb[i](x=h, edge_index=edge_index)
                    # h_k = self.nei_ind_emb[i](graph=to_dgl(Data(edge_index=edge_index, num_nodes=features.shape[0])), feat=h)
                    a = self.nei_rel_learn[i](torch.cat([h_k, h], dim=-1))
                    h = a * h_k + (1 - a) * h
                    if hiddens:
                        hs.append(h.detach().clone())
                return (h, hs) if hiddens else h

            if self.RN == "none":
                if not IN_config.fast or self.nei_feats is None:
                    self.nei_feats, _ = inceptive_aggregation(
                        adj=edge_index,
                        features=features,
                        IN_config=IN_config,
                        preprocess=True,
                        device=device,
                    )
                ns = [self.nei_ind_emb(self.nei_feats[i]) for i in range(self.n_nie)]
                return torch.cat(ns, dim=1)

        if self.IN == "gcn-IN-SN":
            if not IN_config.fast or self.nei_feats is None:
                self.nei_feats, _ = inceptive_aggregation(
                    adj=edge_index,
                    features=features,
                    IN_config=IN_config,
                    preprocess=True,
                    device=device,
                )
            ns = [self.nei_ind_emb[i](self.nei_feats[i]) for i in range(self.n_nie)]

            return {
                "concat": (
                    (self.nei_rel_learn(torch.cat(ns, dim=1)), ns)
                    if hiddens
                    else self.nei_rel_learn(torch.cat(ns, dim=1))
                ),
                "none": torch.cat(ns, dim=1),
            }[self.RN]

        raise ValueError(f"Either IN: {self.IN} or RN: {self.RN} is not valid.")

    def get_Ws(self):
        if self.RN in ["none", "residual"]:
            Ws = [self.nei_ind_emb[0][1].weight.detach().clone()]
            for i in range(1, len(self.nei_ind_emb)):
                Ws.append(self.nei_ind_emb[i].lin.weight.detach().clone())
            return Ws
        if self.RN == "attentive":
            Ws = [self.nei_ind_emb[0][1].weight.detach().clone()]
            for i in range(1, len(self.nei_rel_learn)):
                Ws.append(self.nei_rel_learn[i][1].weight.detach().clone())
            return Ws

        if self.RN == "concat":
            Ws = []
            for i, nei_ind_emb in enumerate(self.nei_ind_emb):
                Ws.append(nei_ind_emb[1].weight.detach().clone())
            return Ws, self.nei_rel_learn[1].weight.detach().clone()

        raise ValueError(f'{self.RN} not in ["none", "residual", "attentive", "concat"]')
