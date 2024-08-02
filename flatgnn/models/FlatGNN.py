"""FlatGNN"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
import copy
import math
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

from ..modules import FlatGNN as FlatGNN_layer
from ..modules import MLP


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


class FlatGNN(nn.Module):
    """FlatGNN"""
    def __init__(
        self,
        in_feats,
        h_feats,
        n_clusters,
        n_epochs=2000,
        lr: float = 0.001,
        l2_coef: float = 0.0001,
        early_stop: int = 100,
        device=None,
        dropout: float = 0.0,
        dropout2: float = 0.0,
        n_hops=6,
        n_intervals=3,
    ) -> None:
        super().__init__()
        self.n_intervals = n_intervals
        self.dropout = dropout
        self.dropout2 = dropout2

        # FlatGNN
        self.n_epochs = n_epochs
        self.h_feats = h_feats
        self.l2_coef = l2_coef

        # URL
        self.lr = lr
        self.estop_steps = early_stop
        self.n_clusters = n_clusters
        self.device = device
        self.best_model = None

        self.set_gnn = FlatGNN_layer(
            in_feats=in_feats,
            h_feats=h_feats,
            n_hops=n_hops,
            dropout=dropout,
            n_intervals=n_intervals,
        )
        self.set_gnn_1 = None
        # self.set_gnn_1 = FlatGNN_layer(
        #     in_feats=h_feats * (n_hops - n_intervals + 2),
        #     h_feats=h_feats,
        #     n_hops=n_hops,
        #     dropout=dropout,
        #     n_intervals=n_intervals,
        #     no_save=True,
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

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.l2_coef,
        )

    def forward(
        self,
        features,
        graph=None,
        device=None,
    ):
        z_set_gnn = self.set_gnn(graph=graph, device=device)
        z_set_gnn = (
            self.set_gnn_1(graph=graph, device=device, feats=torch.cat(z_set_gnn, dim=1))
            if self.set_gnn_1 is not None else z_set_gnn
        )

        return torch.cat(z_set_gnn, dim=1)

    def validate(self, features, labels, train_mask, val_mask, test_mask, graph):
        self.train(False)
        with torch.no_grad():
            embeddings = self.forward(
                features,
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

    def fit(
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
            log_dir=
            f"logs/runs/{get_str_time()[:10]}/joint_{graph.name}_{split_id}_{self.h_feats}_{get_str_time()[11:]}"
        )

        t_start = time.time()
        for epoch in range(self.n_epochs):
            t = time.time()
            self.train()
            self.optimizer.zero_grad()

            embeddings = self.forward(
                graph.ndata["feat"],
                graph=graph,
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

        t_finish = time.time()
        print(f"10 epoch cost: {(t_finish-t_start)/self.n_epochs * 10:.4f}s")
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

    def get_embeddings(self):
        with torch.no_grad():
            pass
