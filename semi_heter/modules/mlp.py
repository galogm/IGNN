"""MLP"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
from typing import Any, Callable, Dict, List, Tuple

import torch.nn.functional as F
from torch import nn


def scale(z):
    """Feature Scale
    Args:
        z (torch.Tensor):hidden embedding

    Returns:
        z_scaled (torch.Tensor):scaled embedding
    """
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / (zmax - zmin)
    z_scaled = z_std
    return z_scaled


class LinTrans(nn.Module):
    """Linear Transform Model

    Args:
        layers (int):number of linear layers.
        dims (list):Number of units in hidden layers.
    """

    def __init__(self, layers, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        """Forward Propagation

        Args:
            x (torch.Tensor):feature embedding

        Returns:
            out (torch.Tensor):hiddin embedding
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        out = scale(out)
        out = F.normalize(out)
        return out


class MLP(nn.Module):
    """Self Reconstruction."""

    def __init__(
        self,
        in_feats,
        h_feats: List[int],
        layers: int = 1,
        acts: List[Callable] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        pl = []
        in_feats = [in_feats] + h_feats
        for i in range(self.layers):
            if acts is not None and len(acts) > 0 and acts[i] is not None:
                pl.append(
                    nn.Sequential(
                        nn.Linear(in_feats[i], h_feats[i]),
                        nn.LayerNorm(h_feats[i]),
                        acts[i],
                    )
                )
            else:
                pl.append(nn.Linear(in_feats, h_feats[i]))
        self.ec = nn.ModuleList(pl)

    def forward(self, x):
        z = x
        for i in range(self.layers):
            z = self.ec[i](
                F.dropout(
                    z,
                    self.dropout,
                    self.training,
                )
            )
        return z


class Res(nn.Module):
    """Self Reconstruction."""

    def __init__(
        self,
        in_feats,
        h_feats: List[int],
        layers: int = 1,
        acts: List[Callable] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        pl = []
        resl = []
        in_feats = [in_feats] + h_feats
        self.acts = acts
        for i in range(self.layers):
            if acts is not None and len(acts) > 0 and acts[i] is not None:
                pl.append(
                    nn.Sequential(
                        nn.Linear(in_feats[i], h_feats[i]),
                        acts[i],
                        nn.Linear(h_feats[i], h_feats[i]),
                        nn.LayerNorm(h_feats[i]),
                    )
                )
                resl.append(
                    nn.Sequential(
                        nn.Linear(in_feats[i], h_feats[i]),
                        nn.LayerNorm(h_feats[i]),
                    )
                )
            else:
                pl.append(nn.Linear(in_feats, h_feats[i]))
        self.ec = nn.ModuleList(pl)
        self.res = nn.ModuleList(resl)

    def forward(self, x):
        z = x
        for i in range(self.layers):
            z = (
                self.acts[i](self.ec[i](z))
                if i == 0
                else self.acts[i](
                    self.ec[i](
                        F.dropout(
                            z,
                            self.dropout,
                            self.training,
                        )
                    )
                    + self.res[i](
                        F.dropout(
                            z,
                            self.dropout,
                            self.training,
                        )
                    )
                )
            )
        return z
