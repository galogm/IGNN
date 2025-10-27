import os
import sys
import time

sys.path.append("../../..")
import numpy as np
import torch
import torch.nn.functional as F
from graph_datasets import load_data

# from .process import *, utils_gcnii
from src import utils_gcnii
from src import wgnn_graphops as GO
from src import wgnn_network as GN
from the_utils import save_to_csv_files, set_device, set_seed, split_train_test_nodes
from torch_geometric.utils import sparse as sparseConvert

from ignn.modules import DataConf
from ignn.utils import read_configs

DATA = DataConf(**read_configs("data"))

import argparse


def get_splits_mask(
    num_nodes,
    name,
    train_ratio,
    valid_ratio,
    repeat,
    split_id,
    SPLIT_DIR,
    mask=True,
):
    train_idx, val_idx, test_idx = split_train_test_nodes(
        num_nodes=num_nodes,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        data_name=name,
        split_id=split_id,
        split_times=repeat,
        fixed_split=True,
        split_save_dir=SPLIT_DIR,
    )
    if not mask:
        return {
            "train": torch.tensor(train_idx, dtype=torch.int64),
            "valid": torch.tensor(val_idx, dtype=torch.int64),
            "test": torch.tensor(test_idx, dtype=torch.int64),
        }
    train_mask = (
        torch.zeros(num_nodes).scatter_(0, torch.tensor(train_idx, dtype=torch.int64), 1).bool()
    )
    val_mask = (
        torch.zeros(num_nodes).scatter_(0, torch.tensor(val_idx, dtype=torch.int64), 1).bool()
    )
    test_mask = (
        torch.zeros(num_nodes).scatter_(0, torch.tensor(test_idx, dtype=torch.int64), 1).bool()
    )

    return train_mask, val_mask, test_mask


parser = argparse.ArgumentParser(description="wgnn_fully_supervised")
parser.add_argument(
    "--dataset",
    default="cora",
    type=str,
    help="dataset name",
)
parser.add_argument(
    "--source",
    default="pyg",
    type=str,
    help="dataset source",
)

parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="seed",
)

parser.add_argument(
    "--runs",
    default=10,
    type=int,
    help="runs",
)

parser.add_argument(
    "--device",
    default=0,
    type=int,
    help="device",
)

parser.add_argument(
    "--omega",
    default=1,
    type=int,
    help="1 if use omegaGCN, 0 otherwise",
)

parser.add_argument(
    "--attspat",
    default=1,
    type=int,
    help="1 if use attention for spatial operation, 0 otherwise",
)


parser.add_argument(
    "--attHeads",
    default=1,
    type=int,
    help="number of attention heads",
)

parser.add_argument(
    "--numOmega",
    default=1,
    type=int,
    help="number of omega to learn",
)

parser.add_argument(
    "--n_channels",
    default=64,
    type=int,
    help="n_channels",
)

parser.add_argument(
    "--nlayers",
    default=2,
    type=int,
    help="n_channels",
)

parser.add_argument(
    "--dropout",
    default=0.5,
    type=float,
    help="dropout",
)
parser.add_argument(
    "--lr",
    default=0.001,
    type=float,
    help="lr",
)
parser.add_argument(
    "--wd",
    default=0.0001,
    type=float,
    help="lr",
)

nomega = 1
nheads = 1
ncheckpoints = 1

args = parser.parse_args()

nlayers = args.nlayers
n_channels = args.n_channels
nopen = n_channels
nhid = n_channels
nNclose = n_channels

nlayer = nlayers
datastr = args.dataset
dropout = args.dropout

lr = args.lr
attLR = args.lr
lrGCN = args.lr
lrOmega = args.lr

wd = args.wd
wdGCN = args.wd
wdOmega = args.wd
attWD = args.wd


def train_step(model, optimizer, features, labels, adj, idx_train):
    model.train()
    optimizer.zero_grad()
    I = adj[0, :]
    J = adj[1, :]
    N = labels.shape[0]
    w = torch.ones(adj.shape[1]).to(device)
    G = GO.graph(I, J, N, W=w, pos=None, faces=None)
    G = G.to(device)
    xn = features
    [out, _] = model(xn, G, omega=args.omega, attention=args.attspat, checkpoints=ncheckpoints)
    acc_train = utils_gcnii.accuracy(out[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(out[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def eval_test_step(model, features, labels, adj, idx_test):
    model.eval()
    with torch.no_grad():
        I = adj[0, :]
        J = adj[1, :]
        N = labels.shape[0]
        w = torch.ones(adj.shape[1]).to(device)

        G = GO.graph(I, J, N, W=w, pos=None, faces=None)
        G = G.to(device)
        xn = features
        [out, _] = model(xn, G, omega=args.omega, attention=args.attspat, checkpoints=ncheckpoints)

        loss_test = F.nll_loss(out[idx_test], labels[idx_test].to(device))
        acc_test = utils_gcnii.accuracy(out[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test.item()


def train(datastr, splitstr, num_output, run):
    slurm = False
    # adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = process.full_load_data(
    #     datastr,
    #     splitstr, slurm=slurm)
    # adj = adj.to_dense()
    # [edge_index, edge_weight] = sparseConvert.dense_to_sparse(adj)
    # del adj

    dataset = load_data(
        dataset_name=args.dataset,
        directory=DATA.DATA_DIR,
        source=args.source,
        row_normalize=True,
        rm_self_loop=False,
        add_self_loop=True,
        to_simple=True,
        verbosity=3,
        return_type="pyg",
    ).to(device)
    args.name = dataset.name
    edge_index = dataset.edge_index.to(torch.int64)
    features = dataset.x
    labels = dataset.y
    num_output = dataset.num_classes

    split_idx = get_splits_mask(
        dataset.num_classes,
        dataset.name,
        48,
        32,
        args.runs,
        run,
        DATA.SPLIT_DIR,
        mask=False,
    )

    # edge_index = edge_index.to(device)
    features = features.to(device).t().unsqueeze(0)
    idx_train = split_idx["train"].to(device)
    idx_valid = split_idx["valid"].to(device)
    idx_test = split_idx["test"].to(device)
    # labels = labels.to(device)

    numAttHeads = args.attHeads if args.attspat else 1
    model = GN.wgnn(
        features.shape[1],
        nopen,
        nhid,
        nlayer,
        num_output=num_output,
        dropOut=dropout,
        numAttHeads=numAttHeads,
        num_omega=nomega,
        omega_perchannel=nomega,
    )
    model.reset_parameters()
    model = model.to(device)
    optimizer = torch.optim.Adam(
        [
            dict(params=model.KN1, lr=lrGCN, weight_decay=wdGCN),
            dict(params=model.K1Nopen, weight_decay=wd),
            dict(params=model.KNclose, weight_decay=wd),
            dict(params=model.att_src, lr=attLR, weight_decay=attWD),
            dict(params=model.att_dst, lr=attLR, weight_decay=attWD),
            dict(params=model.omega, lr=lrOmega, weight_decay=wdOmega),
        ],
        lr=lr,
    )
    bad_counter = 0
    best = 0
    t_s = time.time()
    for epoch in range(1000):
        loss_tra, acc_tra = train_step(model, optimizer, features, labels, edge_index, idx_train)
        loss_val, acc_val = eval_test_step(model, features, labels, edge_index, idx_valid)
        loss_test, acc_test = eval_test_step(model, features, labels, edge_index, idx_test)

        if acc_val > best:
            best = acc_val
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 100:
            break
    acc = acc_test

    return acc * 100, (time.time() - t_s) / (epoch + 1) * 100


set_seed(args.seed)
device = set_device(args.device)

acc_list = []
time_list = []
for i in range(10):
    if datastr == "cora":
        num_output = 7
    elif datastr == "citeseer":
        num_output = 6
    elif datastr == "pubmed":
        num_output = 3
    elif datastr == "chameleon":
        num_output = 5
    else:
        num_output = 5
    splitstr = "../splits/" + datastr + "_split_0.6_0.2_" + str(i) + ".npz"

    acc, t = train(datastr, splitstr, num_output, i)
    acc_list.append(acc)
    time_list.append(t)
    print(i, ": {:.2f}".format(acc_list[-1]))

mean_test_acc = np.mean(acc_list)
std_test_acc = np.std(acc_list)
print(f"Final Test: {mean_test_acc} ± {std_test_acc}")


save_to_csv_files(
    results={"acc": f"{mean_test_acc} ± {std_test_acc}"},
    insert_info={
        "dataset": args.name,
        "model": "wgnn",
    },
    append_info={
        "time": f"{np.mean(time_list):.2f}",
        "args": args.__dict__,
        "source": args.source,
    },
    csv_name=f"baselines.csv",
)
