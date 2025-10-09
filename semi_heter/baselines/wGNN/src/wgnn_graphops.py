import torch
import torch.nn as nn


class graph(nn.Module):
    def __init__(
        self,
        iInd,
        jInd,
        nnodes,
        W=torch.tensor([1.0]),
        pos=None,
        faces=None,
        device=torch.device("cpu"),
    ):
        super(graph, self).__init__()
        self.iInd = iInd.long()
        self.jInd = jInd.long()
        self.nnodes = nnodes
        self.W = W.to(device)
        self.pos = pos
        self.faces = faces

    def neighborNode(self, x, W=[], replaceW=False):
        if len(W) == 0:
            W = self.W
        else:
            if not replaceW:
                W = self.W * W
        if W.shape[0] == x.shape[2]:
            g = W[self.iInd] * (x[:, :, self.jInd])
        else:
            if x.shape[0] == 1:
                g = W * (x[:, :, self.jInd])
            else:
                g = W[self.iInd].t().unsqueeze(1) * (x[:, :, self.jInd])
        return g

    def neighborEdge(self, g, W=[], replaceW=False):
        if len(W) == 0:
            W = self.W
        else:
            if not replaceW:
                W = self.W * W
        x2 = torch.zeros(g.shape[0], g.shape[1], self.nnodes, device=g.device)
        if W.shape[0] != g.shape[2]:
            x2.index_add_(2, self.iInd, W[self.iInd] * g)
        else:
            x2.index_add_(2, self.iInd, W * g)

        x = x2
        return x
