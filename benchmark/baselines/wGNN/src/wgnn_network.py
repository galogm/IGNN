import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import Linear, ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils.softmax import softmax

try:
    from src import wgnn_graphops as GO
    from src.inits import glorot, identityInit

except:
    # import graphOps as GO
    from inits import glorot, identityInit


def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def GINMLP(channels):
    return Seq(
        Linear(channels, channels),
        torch.nn.BatchNorm1d(channels),
        ReLU(),
        Linear(channels, channels),
    )


class wgnn(nn.Module):
    def __init__(
        self,
        nNin,
        nopen,
        nhid,
        nlayer,
        num_output=7,
        dropOut=False,
        ppi=False,
        ogbn=False,
        numAttHeads=1,
        num_omega=1,
        TUDatasets=False,
        omega_perchannel=True,
        singleOmega=False,
    ):
        super(wgnn, self).__init__()
        self.singleOmega = singleOmega
        self.omega_perchannel = omega_perchannel
        self.ppi = ppi
        self.TUDatasets = TUDatasets
        self.ogbn = ogbn
        if TUDatasets:
            self.convs = torch.nn.ModuleList()
            for i in range(nlayer):
                self.convs.append(GINMLP(nopen))
            self.lins = torch.nn.ModuleList()
            self.lins.append(Linear(nopen, nopen))
            self.lins.append(Linear(nopen, num_output))

        if self.ppi or self.ogbn:
            self.BNs = nn.ModuleList()
            for i in range(nlayer):
                self.BNs.append(nn.BatchNorm1d(nopen))

        self.num_output = num_output
        if dropOut > 0.0:
            self.dropout = dropOut
        else:
            self.dropout = False
        self.nlayers = nlayer
        stdv = 1e-2
        stdvp = 1e-4
        self.K1Nopen = nn.Parameter(torch.randn(nopen, nNin) * stdv)
        self.nopen = nopen
        self.KNclose = nn.Parameter(
            torch.randn(num_output, self.nopen) * stdv
        )  # num_output on left size

        Nfeatures = 1 * nopen
        self.KN1 = nn.Parameter(torch.rand(nlayer, Nfeatures, nhid) * stdvp)
        self.KN1 = nn.Parameter(
            identityInit(self.KN1) + torch.rand(nlayer, Nfeatures, nhid) * stdvp
        )

        self.numAttHeads = numAttHeads
        # The learnable parameters to compute attention coefficients:
        self.att_src = nn.Parameter(
            1e-3 * torch.ones(nlayer, self.numAttHeads, self.nopen // self.numAttHeads)
        )
        self.att_dst = nn.Parameter(
            1e-3 * torch.ones(nlayer, self.numAttHeads, self.nopen // self.numAttHeads)
        )

        self.nomega = num_omega
        self.multi = max(1, self.numAttHeads)

        self.omega = nn.Parameter(torch.ones(nlayer, self.nomega, self.multi))
        if self.omega_perchannel:
            self.omega = nn.Parameter(torch.ones(nlayer, self.nopen))
        else:
            self.omega = nn.Parameter(torch.ones(nlayer))
            self.omega = nn.Parameter(torch.ones(nlayer))

    def reset_parameters(self):
        glorot(self.K1Nopen)
        glorot(self.KNclose)
        glorot(self.att_src)
        glorot(self.att_dst)

    def edgeConv(self, xe, K, groups=1):
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1), groups=groups)
            else:
                xe = conv2(xe, K, groups=groups)
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1), groups=groups)
            else:
                xe = conv1(xe, K, groups=groups)
        return xe

    def singleLayer(
        self, x, K, relu=True, norm=False, groups=1, openclose=False, bn=None, act=True
    ):
        if openclose:  # if K.shape[0] != K.shape[1]:
            x = self.edgeConv(x, K)
            if norm:
                x = F.instance_norm(x)
            if bn is not None:
                x = bn(x)
            if act:
                if relu:
                    x = F.relu(x)
                else:
                    x = F.tanh(x)
        if not openclose:  # if K.shape[0] == K.shape[1]:
            x = self.edgeConv(x, K)
            if act:
                if not relu:
                    x = F.tanh(x)
                else:
                    x = F.relu(x)
            if norm:
                x = F.layer_norm(x, x.shape)
                beta = torch.norm(x)
            x = self.edgeConv(x, K.t())
        return x

    def updateGraph(
        self, Graph, features=None, selfLoops=True, improved=True, nnodes=None, device=torch.cpu
    ):
        # If features are given - update graph according to feaure space l2 distance
        N = Graph.nnodes
        I = Graph.iInd
        J = Graph.jInd
        edge_index = torch.cat([I.unsqueeze(0), J.unsqueeze(0)], dim=0)
        if features is not None:
            features = features.squeeze()
            D = torch.relu(
                torch.sum(features**2, dim=0, keepdim=True)
                + torch.sum(features**2, dim=0, keepdim=True).t()
                - 2 * features.t() @ features
            )
            D = D / D.std()
            D = torch.exp(-2 * D)
            w = D[I, J]
            Graph = GO.graph(I, J, N, W=w, pos=None, faces=None, device=device)

        else:
            [edge_index, edge_weights] = gcn_norm(
                edge_index, add_self_loops=selfLoops, improved=improved
            )
            I = edge_index[0, :]
            J = edge_index[1, :]
            Graph = GO.graph(I, J, N, W=edge_weights, pos=None, faces=None, device=device)

        return Graph, edge_index

    def attCoeffs(self, xn, att_src, att_dst, I, J, use_softmax=True):
        nnodes = xn.shape[-1]
        if att_src.shape[0] == 1:
            W = (att_src * xn[:, :, I].squeeze().t()).sum(-1) + (
                att_dst * xn[:, :, J].squeeze().t()
            ).sum(-1)
        else:
            W = (att_src @ xn[:, :, I]).squeeze().transpose(-1, 0).sum(-1) + (
                att_dst @ xn[:, :, J]
            ).squeeze().transpose(-1, 0).sum(-1)

        W = F.leaky_relu(W, 0.2)
        if use_softmax:
            W = softmax(W, I, num_nodes=nnodes)
        return W

    def run_function(self, start, end):
        def custom_forward(*inputs):
            xn, Graph, omega, attention = inputs
            for i in range(start, end):
                if attention:
                    H = self.numAttHeads
                    C = self.nopen // self.numAttHeads
                    xn = xn.view(H, C, -1)
                    W = self.attCoeffs(
                        xn,
                        self.att_src[i, :, :],
                        self.att_dst[i, :, :],
                        Graph.iInd,
                        Graph.jInd,
                        use_softmax=True,
                    )
                    replace = True
                else:
                    W = []
                    replace = False
                message = Graph.neighborNode(xn, W, replaceW=replace)
                spatial = Graph.neighborEdge(
                    message, W=torch.ones(xn.shape[-1], device=xn.device), replaceW=True
                )
                # Apply spatial operation:
                if omega:
                    if attention:
                        xn = xn.view(-1, xn.shape[-1]).unsqueeze(0)
                        spatial = spatial.view(-1, spatial.shape[-1]).unsqueeze(0)
                    if not self.singleOmega:
                        xn = xn - (self.omega[i].view(1, -1, 1)) * (xn - spatial)
                    else:
                        xn = xn - (self.omega[0].view(1, -1, 1)) * (xn - spatial)
                else:
                    xn = spatial
                # Apply 1x1 conv:
                if self.TUDatasets:
                    xn = xn.squeeze().t()
                    xn = self.convs[i](xn)
                    xn = F.relu(xn)
                    xn = xn.t().unsqueeze(0)
                else:
                    xn = self.singleLayer(
                        xn,
                        self.KN1[i],
                        norm=False,
                        relu=True,
                        groups=1,
                        bn=None,
                        openclose=True,
                        act=False,
                    )
                    if self.ppi or self.ogbn:
                        xn = self.BNs[i](xn)
                    else:
                        xn = F.relu(xn)

            return xn

        return custom_forward

    def forward(self, xn, Graph, data=None, omega=True, attention=True, checkpoints=1):
        # Opening layer
        if not self.TUDatasets:
            [Graph, _] = self.updateGraph(
                Graph, selfLoops=True, nnodes=xn.shape[-1], improved=False, device=xn.device
            )
        if self.dropout:
            xn = F.dropout(xn, p=self.dropout, training=self.training)
        xn = self.singleLayer(xn, self.K1Nopen, relu=True, openclose=True, norm=False)
        nlayers = self.nlayers

        segment_size = nlayers // checkpoints
        for start in range(0, segment_size * (checkpoints), segment_size):
            end = start + segment_size
            xn = checkpoint.checkpoint(self.run_function(start, end), xn, Graph, omega, attention)

        # Closing layer:
        if not self.TUDatasets:
            if self.dropout:
                xn = F.dropout(xn, p=self.dropout, training=self.training)
            xn = F.conv1d(xn, self.KNclose.unsqueeze(-1))
            xn = xn.squeeze().t()
            if not self.ppi:
                return F.log_softmax(xn, dim=-1), Graph
            else:
                return xn, Graph
        else:
            xn = xn.squeeze().t()
            xn = global_add_pool(xn, data.batch)
            xn = self.lins[0](xn).relu()
            xn = F.dropout(xn, self.dropout, training=self.training)
            xn = self.lins[1](xn)
            return F.log_softmax(xn, dim=-1)
