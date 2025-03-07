"""IGNN"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals,invalid-name,too-many-branches,too-many-statements
import copy
import time
import traceback
from typing import Literal, Optional

import torch
import torch_geometric.transforms as T
from the_utils import get_str_time, save_model
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..modules import IGNNConv, INConf
from ..utils import metric

losses = {
    "ce": torch.nn.CrossEntropyLoss,
    "bce": torch.nn.BCEWithLogitsLoss,
}


class IGNN(nn.Module):
    """IGNN"""

    def __init__(
        self,
        in_feats,
        h_feats,
        n_clusters,
        n_epochs=2000,
        lr: float = 0.001,
        l2_coef: float = 0.00005,
        early_stop: int = 100,
        device=None,
        nas_dropout: float = 0.0,
        nss_dropout: float = 0.8,
        clf_dropout: float = 0.9,
        out_ndim_trans: int = 64,
        n_hops=6,
        IN="gcn",
        RN="concat",
        n_layers=1,
        act="relu",
        norm: Optional[Literal["bn", "ln"]] = None,
        loss="ce",
        num_heads=1,
        transform_first=False,
    ) -> None:
        super().__init__()

        # pylint: disable=consider-iterating-dictionary
        assert loss in losses.keys(), f"loss should be in {losses.keys()}"

        self.nss_dropout = nss_dropout
        self.n_epochs = n_epochs
        self.h_feats = h_feats
        self.l2_coef = l2_coef
        self.lr = lr
        self.estop_steps = early_stop
        self.n_clusters = n_clusters
        self.device = device
        self.best_model = None

        self.ignnconvs = nn.ModuleList(
            [
                IGNNConv(
                    IN=IN,
                    in_feats=in_feats if i == 0 else h_feats,
                    h_feats=h_feats,
                    act=act,
                    nas_dropout=nas_dropout,
                    nss_dropout=nss_dropout,
                    norm=norm,
                    transform_first=transform_first,
                    RN=RN,
                    n_hops=n_hops,
                    ndim_fc=h_feats * (n_hops + 1),
                )
                for i in range(n_layers)
            ]
        )

        hidden_dim = {
            "multi-con": max(h_feats * (n_hops - 1), h_feats),
            "concat": h_feats,
            "ordered-gating": h_feats,
            "max": h_feats,
            "mean": h_feats,
            "sum": h_feats,
            "lstm": h_feats,
            "none": h_feats,
            "residual": h_feats,
            "attentive": h_feats,
        }[RN]

        self.classifier = nn.Sequential(
            nn.Dropout(p=clf_dropout), nn.Linear(hidden_dim, n_clusters)
        )

        self.criterion = losses[loss]()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)

    def validate(
        self,
        edge_index,
        features,
        labels,
        IN_config: INConf,
        epoch,
        eval_interval=1,
        eval_start=0,
        test_loader=None,
        train_mask=None,
        val_mask=None,
        test_mask=None,
    ):
        with torch.no_grad():
            self.train(False)

            try:
                if epoch % eval_interval == 0 and epoch >= eval_start:
                    if test_loader is not None:
                        device = (
                            torch.device("cpu")
                            if IN_config.name in ["products_ogb", "pokec_linkx"]
                            and len(test_loader) == 1
                            else self.device
                        )
                        transform = T.Compose([T.ToDevice(device), T.ToSparseTensor()])
                        y_true = {"train": [], "val": [], "test": []}
                        y_pred = {"train": [], "val": [], "test": []}
                        for data in tqdm(test_loader, "val step"):
                            data = transform(data)
                            embeddings = self.forward(
                                edge_index=data.adj_t,
                                features=data.x,
                                IN_config=IN_config,
                                device=device,
                                fast=False,
                            )
                            logits = self.classifier(embeddings.to(self.device))
                            for split in ["train", "val", "test"]:
                                mask = data[f"{split}_mask"]
                                y_true[split].append(data.y[mask].detach().cpu())
                                y_pred[split].append(logits[mask].detach().cpu())

                        train_acc = metric(
                            IN_config.name,
                            logits=torch.cat(y_pred["train"], dim=0),
                            labels=torch.cat(y_true["train"], dim=0),
                        )
                        valid_acc = metric(
                            IN_config.name,
                            logits=torch.cat(y_pred["val"], dim=0),
                            labels=torch.cat(y_true["val"], dim=0),
                        )
                        test_acc = metric(
                            IN_config.name,
                            logits=torch.cat(y_pred["test"], dim=0),
                            labels=torch.cat(y_true["test"], dim=0),
                        )
                        return train_acc, valid_acc, test_acc

                    embeddings = self.forward(edge_index, features, IN_config, device=self.device)
                    logits = self.classifier(embeddings)
                    return metric(IN_config.name, logits, labels, train_mask, val_mask, test_mask)
                return 0, 0, 0

            except RuntimeError:
                traceback.print_exc()
                self.save_model(IN_config.name, epoch, None)
                return 0, 0, 0

    def forward(self, edge_index, features, IN_config: INConf, device=None, fast=None):
        H = None
        for i, ignnconv in enumerate(self.ignnconvs):
            H = ignnconv(
                edge_index,
                features=features if i == 0 else H,
                IN_config=INConf(
                    **{**IN_config.__dict__, "fast": i == 0 if fast is None else fast}
                ),
                device=device,
            )
        return H

    def fit(
        self,
        edge_index,
        features,
        labels,
        IN_config: INConf,
        train_loader=None,
        test_loader=None,
        train_mask=None,
        val_mask=None,
        test_mask=None,
        eval_interval=1,
        eval_start=0,
        save_state=False,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.to(self.device)
        transform = T.Compose(
            [T.ToDevice(device if not IN_config.fast else torch.device("cpu")), T.ToSparseTensor()]
        )
        labels = labels.to(self.device)

        best_epoch = 0
        best_acc = best_acc_t = 0.0
        cnt = 0
        best_state_dict = None
        # writer = SummaryWriter(
        #     log_dir=f"logs/runs/{get_str_time()[:10]}/joint_{name}_{split_id}_{self.h_feats}_{get_str_time()[11:]}"
        # )

        t_start = time.time()
        for epoch in range(self.n_epochs):
            loss_value = 0
            # loss_val_value = 0
            # loss_test_value = 0

            self.train()
            if train_loader is not None:
                for data in tqdm(train_loader, "train step"):
                    data = transform(data)
                    self.optimizer.zero_grad()

                    embeddings = self.forward(data.adj_t, data.x, IN_config, device, fast=False)
                    logits = self.classifier(embeddings)

                    loss = self.criterion(logits[data.train_mask], data.y[data.train_mask])
                    # loss_val = self.criterion(logits[data.val_mask], data.y[data.val_mask])
                    # loss_test = self.criterion(logits[data.test_mask], data.y[data.test_mask])
                    loss_value += loss.item()
                    # loss_val_value += loss_val.item()
                    # loss_test_value += loss_test.item()

                    loss.backward()
                    self.optimizer.step()

                train_acc, valid_acc, test_acc = self.validate(
                    edge_index,
                    features,
                    labels,
                    IN_config,
                    epoch,
                    eval_interval=eval_interval,
                    eval_start=eval_start,
                    test_loader=test_loader,
                )
            else:
                self.optimizer.zero_grad()

                embeddings = self.forward(edge_index, features, IN_config, device=device)
                logits = self.classifier(embeddings)

                loss = self.criterion(logits[train_mask], labels[train_mask])
                # loss_val = self.criterion(logits[val_mask], labels[val_mask])
                # loss_test = self.criterion(logits[test_mask], labels[test_mask])
                loss_value = loss.item()
                # loss_val_value = loss_val.item()
                # loss_test_value = loss_test.item()

                loss.backward()
                self.optimizer.step()

                train_acc, valid_acc, test_acc = self.validate(
                    edge_index,
                    features,
                    labels,
                    IN_config,
                    epoch,
                    eval_interval=eval_interval,
                    eval_start=eval_start,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                )

            if valid_acc == 0 and best_acc == 0:
                print(f"Epoch {epoch} done.")
            else:
                if valid_acc >= best_acc:
                    cnt = 0
                    best_acc = valid_acc
                    best_acc_t = test_acc
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(self.state_dict())
                else:
                    cnt += 1

                print(
                    f"epoch:{epoch}/best:{best_epoch}, loss:{loss_value:.4f}, train:{train_acc:.4f}, "
                    f"valid:{valid_acc:.4f}/best:{best_acc:.4f}, test:{test_acc:.4f}/best:{best_acc_t:.4f}"
                )

            if cnt == self.estop_steps:
                print(
                    f"Early Stopping! Best Epoch: {best_epoch}, "
                    f"best val acc: {best_acc:.4f}, test acc: {best_acc_t:.4f}"
                )
                break

            # writer.add_scalar("joint/time/train", time.time() - t, epoch)

            # writer.add_scalar("joint/metric/train", train_acc, epoch)
            # writer.add_scalar("joint/metric/val", valid_acc, epoch)
            # writer.add_scalar("joint/metric/test", test_acc, epoch)
            # writer.add_scalar("joint/loss/train", loss.item(), epoch)
            # writer.add_scalar("joint/loss/val", loss_val.item(), epoch)
            # writer.add_scalar("joint/loss/test", loss_test.item(), epoch)

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
        if save_state:
            self.save_model(IN_config.name, epoch, loss)
        tm = (time.time() - t_start) / max(1, epoch) * 10
        print(f"10 epoch cost: {tm:.4f}s.")
        return tm

    def save_model(self, name, epoch, loss):
        file = f"{name}.{get_str_time()}"
        save_model(file, self, self.optimizer, epoch, loss)
        print(f"Save best state dict to {file}.")
