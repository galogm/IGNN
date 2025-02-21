"""FaltGNN Layer."""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch import nn
from torch.nn import LayerNorm, Linear, ModuleList
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

from ..utils import preprocess_neighborhoods
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
    """IGNN Conv.
    """
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
                ) for _ in range(N_NIE)
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
                    ) if i != 0 else MLP(
                        in_feats=in_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nas_dropout,
                        layer_norm=True,
                    )
                ) for i in range(N_NIE)
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
                    ) for _ in range(N_NIE)
                )
            else:
                self.nei_ind_emb = nn.ModuleList(
                    MLP(
                        in_feats=in_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nas_dropout,
                        layer_norm=layer_norm,
                    ) for _ in range(N_NIE)
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
                    ) if i != 0 else MLP(
                        in_feats=in_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nas_dropout,
                        layer_norm=layer_norm,
                    )
                ) for i in range(N_NIE)
            )
        elif IN == "gcn-nIN-nSN":
            self.nei_ind_emb = nn.ModuleList(
                (
                    GraphConv(
                        in_feats=h_feats,
                        out_feats=h_feats,
                        activation=acts[act](),
                    ) if i != 0 else MLP(
                        in_feats=in_feats,
                        h_feats=[h_feats],
                        acts=[acts[act]()],
                        dropout=nas_dropout,
                        layer_norm=layer_norm,
                    )
                ) for i in range(N_NIE)
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
                    ) for i in range(N_NIE)
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
                        "gcn-IN-SN": True,
                        "gcn-IN-nSN": True,
                    }[self.IN],
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
                        ) if self.adj is None else self.adj
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
                                ) if batch_idx is not None else self.
                                nei_uni_emb(self.nei_feats[i].to(self.device))
                            )
                        )
                    else:
                        self.ns.append(
                            # self.nei_ind_emb[i](self.nei_feats[i].index_select(0, batch_idx))
                            self.nei_ind_emb[i](
                                self.nei_feats[i].index_select(0, batch_idx).to(self.device)
                                if batch_idx is not None else self.nei_feats[i].to(self.device)
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
            return self.nei_rel_leaRN[0](torch.max(
                torch.stack(self.ns, dim=1),
                dim=1,
            )[0])
        if self.RN == "mean":
            return self.nei_rel_leaRN[0](torch.mean(
                torch.stack(self.ns, dim=1),
                dim=1,
            ))
        if self.RN == "sum":
            return self.nei_rel_leaRN[0](torch.sum(
                torch.stack(self.ns, dim=1),
                dim=1,
            ))
        if self.RN == "lstm":
            return self.nei_rel_leaRN[0](torch.stack(self.ns, dim=1), )
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
                ) for i in range(ignn_layer_num)
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
