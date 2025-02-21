"""FaltGNN Layer."""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
import os
from pathlib import Path

import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from the_utils import make_parent_dirs
from torch import nn
from torch.nn import LayerNorm, Linear, ModuleList
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

from .LSTM import LSTM
from .MLP import MLP
from .OrderedGating import ONGNNConv

acts = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "sigmoid": nn.Sigmoid,
    "none": nn.Identity,
    "tanh": nn.Tanh,
}


class IGNNConv(nn.Module):
    """IGNN Conv."""

    def __init__(
        self,
        IN,
        in_feats,
        h_feats,
        act,
        nas_dropout,
        layer_norm,
        N_NIE,
        transform_first,
        nss_dropout,
        RN,
        n_hops,
        ndim_fc,
        no_save,
    ):
        self.no_save = no_save
        self.n_hops = n_hops
        self.h_feats = h_feats
        self.IN = IN
        self.RN = RN
        self.device = None
        self.transform_first = transform_first
        self.adj = None
        self.adj_und = None
        self.a_feat = None
        self.nss_dropout = nss_dropout
        self.nas_dropout = nas_dropout

        N_NIE = self.n_hops + 1
        super().__init__()
        if IN == "gat":
            self.nei_ind_emb = nn.ModuleList(
                GATConv(
                    in_channels=in_feats,
                    out_channels=h_feats,
                    heads=1,
                    dropout=nas_dropout,
                )
                for _ in range(N_NIE)
            )
        elif IN == "gcn":
            self.nei_ind_emb = nn.ModuleList(
                (
                    GraphConv(
                        in_feats=h_feats,
                        out_feats=h_feats,
                        weight=False if self.RN == "attentive" else True,
                        allow_zero_in_degree=True if self.RN == "attentive" else False,
                        activation=acts[act](),
                    )
                    if i != 0
                    else MLP(
                        in_feats=in_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nas_dropout,
                        layer_norm=True,
                    )
                )
                for i in range(N_NIE)
            )

        elif IN == "gcn-IN-SN":
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
        elif IN == "gcn-IN-nSN":
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
        elif IN == "gcn-nIN-SN":
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
        elif IN == "gcn-nIN-nSN":
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

        if RN == "concat":
            self.nei_rel_leaRN = nn.ModuleList(
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

        elif RN == "residual":
            self.nei_rel_leaRN = nn.ModuleList(
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

        elif RN == "attentive":
            self.nei_rel_leaRN = nn.ModuleList(
                [
                    MLP(
                        in_feats=h_feats * 2,
                        h_feats=[1],
                        acts=[acts["tanh"]()],
                        dropout=nss_dropout,
                        layer_norm=layer_norm,
                    )
                    for i in range(N_NIE)
                ],
            )
        elif RN in ["max", "sum", "mean"]:
            self.nei_rel_leaRN = nn.ModuleList(
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
        elif RN == "lstm":
            self.nei_rel_leaRN = nn.ModuleList(
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

    @staticmethod
    def preprocess_neighborhoods(
        adj: SparseTensor,
        features: torch.FloatTensor,
        name: str,
        n_hops: int,
        set_diag=True,
        remove_diag=False,
        symm_norm=False,
        row_normalized=True,
        device: torch.device = torch.device("cpu"),
        no_save=False,
        return_adj=False,
        process_adj=True,
        save_dir: str = "tmp/ignn/neighborhoods",
        batch_idx=None,
    ):
        sym = "sym" if symm_norm else "nsy"
        diag = "diag" if set_diag else "ndiag"
        diag = "ndiag" if remove_diag else diag
        norm = "norm" if row_normalized else "nnorm"
        base = Path(f"{save_dir}/{name}/{norm}/{diag}/{sym}")

        if process_adj:
            if set_diag:
                print("... setting diagonal entries")
                adj = adj.set_diag()
            elif remove_diag:
                print("... removing diagonal entries")
                adj = adj.remove_diag()
            else:
                print("... keeping diag elements as they are")
            if symm_norm:
                print("... performing symmetric normalization")
                deg = adj.sum(dim=1).to(torch.float)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
                adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
            else:
                print("... performing asymmetric normalization")
                deg = adj.sum(dim=1).to(torch.float)
                deg_inv = deg.pow(-1.0)
                deg_inv[deg_inv == float("inf")] = 0
                adj = deg_inv.view(-1, 1) * adj

        nei_feats = [
            (
                features.to(device).index_select(0, batch_idx)
                if batch_idx is not None
                else features.to(device)
            )
        ]

        # c-IGNN
        if no_save:
            adj = adj.to_torch_sparse_csr_tensor() if process_adj else adj
            for i in range(1, n_hops + 1):
                x = torch.mm(adj, features.cpu())
                nei_feats.append(x.to(torch.float).to(device))
            if return_adj:
                return nei_feats, adj
            return nei_feats

        # Fast c-IGNN
        adj = adj.to_scipy(layout="csr")
        x = features.cpu().numpy()
        print(f"Load aggregated feats from {base}")
        hops = os.listdir(base) if base.exists() else []
        if f"{n_hops}" not in hops:
            for i in range(1, n_hops + 1):
                file = base.joinpath(f"{i}")
                if file.exists():
                    x = torch.load(file, map_location=device)
                    x = x.cpu().numpy()
                else:
                    make_parent_dirs(file)
                    x = adj @ x
                    torch.save(
                        obj=torch.from_numpy(x).to(torch.float).to(device),
                        f=file,
                        pickle_protocol=4,
                    )

        for i in range(1, n_hops + 1):
            file = base.joinpath(f"{i}")
            x = torch.load(file, map_location=device)
            nei_feats.append(x.index_select(0, batch_idx) if batch_idx is not None else x)

        return nei_feats

    def forward(
        self,
        device,
        graph,
        feats=None,
        batch_idx=None,
    ):
        if self.device is None:
            self.to(device=device)
            self.device = device
        self.ns = []
        # adj = graph.adj()
        features = graph.ndata["feat"] if feats is None else feats
        row, col = graph.add_self_loop().edges()

        if self.IN == "gcn" and self.RN == "residual":
            h = features.to(device)
            h = self.nei_ind_emb[0](h)
            for i in range(1, self.n_hops + 1):
                h = self.nei_ind_emb[i](graph=graph.to(device), feat=h) + h
        elif self.IN == "gcn" and self.RN == "attentive":
            h = features.to(device)
            # h = self.nei_ind_emb[0](F.dropout(h,p=self.nas_dropout,training=self.training))
            h = self.nei_ind_emb[0](h)
            for i in range(1, self.n_hops + 1):
                h_k = self.nei_ind_emb[i](graph=graph.to(device), feat=h)
                a = self.nei_rel_leaRN[i](torch.cat([h_k, h], dim=-1))
                h = a * h_k + (1 - a) * h
        elif self.IN == "gcn-nIN-SN":
            h = features.to(device)
            self.ns.append(self.nei_ind_emb[0](h))
            for i in range(1, self.n_hops + 1):
                h = self.nei_ind_emb[i](graph=graph.to(device), feat=h)
                self.ns.append(h)
        elif self.IN == "gcn-nIN-nSN":
            h = self.nei_ind_emb[0](features.to(device))
            self.ns.append(h)
            for i in range(1, self.n_hops + 1):
                h = self.nei_ind_emb[i](graph=graph.to(device), feat=h)
                self.ns.append(h)
        elif self.IN in ["gcn-IN-SN", "gcn-IN-nSN"]:
            n_nodes = graph.num_nodes()
            if (self.nei_feats is None) and not self.no_save:
                if self.adj_und is None:
                    row, col = graph.edges()
                    self.adj_und = SparseTensor(
                        row=row.long(),
                        col=col.long(),
                        sparse_sizes=(n_nodes, n_nodes),
                    )
                self.nei_feats = self.preprocess_neighborhoods(
                    adj=self.adj_und,
                    features=features,
                    row_normalized=graph.row_normalized,
                    name=graph.name,
                    n_hops=self.n_hops,
                    set_diag=True,
                    remove_diag=False,
                    symm_norm={
                        "gcn": True,
                        "gcn-IN-SN": True,
                        "gcn-IN-nSN": True,
                    }[self.IN],
                    device=torch.device("cpu"),
                    batch_idx=None,
                )
            elif self.no_save:
                self.nei_feats, self.adj = self.preprocess_neighborhoods(
                    adj=(
                        SparseTensor(
                            row=row.long(),
                            col=col.long(),
                            sparse_sizes=(n_nodes, n_nodes),
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
                        "gcn-IN-SN": True,
                        "gcn-IN-nSN": True,
                    }[self.IN],
                    device=device,
                    no_save=True,
                    return_adj=True,
                    process_adj=False if self.adj is not None else True,
                    batch_idx=batch_idx,
                )
            if self.IN in ["gcn-IN-SN"]:
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
            elif self.IN in ["gcn-IN-nSN"]:
                for i in range(self.n_hops + 1):
                    self.ns.append(self.nei_ind_emb[0](self.nei_feats[i]))

        if self.RN == "none":
            return torch.cat(self.ns, dim=1)
        if self.RN in ["residual", "attentive"]:
            return h
        if self.RN == "concat":
            hops_feats = torch.cat(self.ns, dim=1)
            return torch.cat(
                [nei_rel_leaRN(hops_feats) for nei_rel_leaRN in self.nei_rel_leaRN],
                dim=-1,
            )
        if self.RN == "max":
            return self.nei_rel_leaRN[0](
                torch.max(
                    torch.stack(self.ns, dim=1),
                    dim=1,
                )[0]
            )
        if self.RN == "mean":
            return self.nei_rel_leaRN[0](
                torch.mean(
                    torch.stack(self.ns, dim=1),
                    dim=1,
                )
            )
        if self.RN == "sum":
            return self.nei_rel_leaRN[0](
                torch.sum(
                    torch.stack(self.ns, dim=1),
                    dim=1,
                )
            )
        if self.RN == "lstm":
            return self.nei_rel_leaRN[0](
                torch.stack(self.ns, dim=1),
            )
        if self.RN == "ordered-gating":
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
            return h


class IGNN_layer(nn.Module):
    """IGNN_layer."""

    def __init__(
        self,
        in_feats,
        h_feats,
        n_hops,
        nas_dropout=0.0,
        nss_dropout=0.8,
        out_ndim_trans=64,
        no_save=False,
        IN="gcn-IN-SN",
        RN="concat",
        act="relu",
        layer_norm=True,
        num_heads=1,
        transform_first=False,
        ignn_layer_num=1,
    ):
        super().__init__()
        self.no_save = no_save
        self.n_hops = n_hops
        self.h_feats = h_feats
        self.IN = IN
        self.device = None
        self.transform_first = transform_first
        self.num_heads = num_heads
        self.out_ndim_trans = out_ndim_trans

        N_NIE = self.n_hops + 1

        ndim_fc = h_feats * N_NIE

        self.ignnconvs = nn.ModuleList(
            [
                IGNNConv(
                    IN,
                    in_feats if i == 0 else h_feats,
                    h_feats,
                    act,
                    nas_dropout,
                    layer_norm,
                    N_NIE,
                    transform_first,
                    nss_dropout,
                    RN=RN,
                    n_hops=n_hops,
                    ndim_fc=ndim_fc,
                    no_save=i != 0,
                )
                for i in range(ignn_layer_num)
            ]
        )

        self.in_ndim_trans = {
            "concat": h_feats,
            "residual": h_feats,
            "attentive": h_feats,
        }[RN]

    def forward(
        self,
        graph: dgl.DGLGraph,
        device: torch.device,
        feats=None,
        batch_idx=None,
    ):
        H = None
        for i, ignnconv in enumerate(self.ignnconvs):
            H = ignnconv(
                device=device,
                graph=graph,
                feats=feats if i == 0 else H,
                batch_idx=batch_idx,
            )

        return H
