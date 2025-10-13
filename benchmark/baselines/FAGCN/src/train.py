import argparse
import os
import sys
import time

sys.path.append('../../..')

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

from .model import FAGCN
from .utils import accuracy, preprocess_data

# torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='chameleon')
parser.add_argument('--source', default='critical')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--device', default=0)
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
# parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
parser.add_argument('--patience', type=int, default=200, help='Patience')
args = parser.parse_args()


from the_utils import (save_to_csv_files, set_device, set_seed,
                       split_train_test_nodes)

set_seed(args.seed)
device = set_device(args.device)
from graph_datasets import load_data

from ignn.modules import DataConf
from ignn.utils import read_configs


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
        return {"train": torch.tensor(train_idx, dtype=torch.int64), "valid": torch.tensor(val_idx, dtype=torch.int64), "test": torch.tensor(test_idx, dtype=torch.int64)}
    train_mask = (
        torch.zeros(num_nodes)
        .scatter_(0, torch.tensor(train_idx, dtype=torch.int64), 1)
        .bool()
    )
    val_mask = (
        torch.zeros(num_nodes)
        .scatter_(0, torch.tensor(val_idx, dtype=torch.int64), 1)
        .bool()
    )
    test_mask = (
        torch.zeros(num_nodes)
        .scatter_(0, torch.tensor(test_idx, dtype=torch.int64), 1)
        .bool()
    )

    return train_mask, val_mask, test_mask


DATA = DataConf(**read_configs("data"))

g, labels, nclass = load_data(
    dataset_name=args.dataset,
    directory=DATA.DATA_DIR,
    source=args.source,
    row_normalize=True,
    rm_self_loop=False,
    add_self_loop=True,
    to_simple=True,
    verbosity=3,
    return_type="dgl",
)
# g.edge_index = dataset.edge_index.to(torch.int64)

features=g.ndata['feat']

features = features.to(device)
labels = labels.to(device)

test_accs = []
times = []
for run in range(args.runs):
    t_s = time.time()

    split_idx = get_splits_mask(
            g.num_nodes(),
            g.name,
            48,
            32,
            args.runs,
            run,
            DATA.SPLIT_DIR,
            mask=False,
        )

    train = split_idx['train'].to(device)
    test = split_idx['test'].to(device)
    val = split_idx['valid'].to(device)

    g = g.to(device)
    deg = g.in_degrees().cuda().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g.ndata['d'] = norm

    net = FAGCN(g, features.size()[1], args.hidden, nclass, args.dropout, args.eps, args.layer_num).cuda()

    # create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # main loop
    dur = []
    los = []
    loc = []
    counter = 0
    min_loss = 100.0
    max_acc = 0.0

    for epoch in range(args.epochs):
        # if epoch >= 3:
        #     t0 = time.time()

        net.train()
        logp = net(features)

        cla_loss = F.nll_loss(logp[train], labels[train])
        loss = cla_loss
        train_acc = accuracy(logp[train], labels[train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        logp = net(features)
        test_acc = accuracy(logp[test], labels[test])
        loss_val = F.nll_loss(logp[val], labels[val]).item()
        val_acc = accuracy(logp[val], labels[val])
        los.append([epoch, loss_val, val_acc, test_acc])

        if loss_val < min_loss and max_acc < val_acc:
            min_loss = loss_val
            max_acc = val_acc
            counter = 0
        else:
            counter += 1


        print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(
            epoch, loss_val, train_acc, val_acc, test_acc))

        if counter >= args.patience:
            print('early stop')
            break

        # if epoch >= 3:
        #     dur.append(time.time() - t0)

    test_accs.append(test_acc)
    times.append((time.time()-t_s)/(epoch+1)*100)

test_accs = np.array(test_accs)
times = np.array(times)

print('Final Test: ',f"{test_accs.mean()} ± {test_accs.std()}")
print('Final Time: ',f"{times.mean():.2f}")

save_to_csv_files(
    results={"acc":f"{test_accs.mean()} ± {test_accs.std()}"},
    insert_info={
        "dataset": g.name,
        "model": "FAGCN",
    },
    append_info={
        "time": f"{times.mean():.2f}",
        "args": args.__dict__,
        "source": args.source,
    },
    csv_name=f"baselines.csv",
)


# if args.dataset in ['cora', 'citeseer', 'pubmed'] or 'syn' in args.dataset:
#     los.sort(key=lambda x: x[1])
#     acc = los[0][-1]
#     print(acc)
# else:
#     los.sort(key=lambda x: -x[2])
#     acc = los[0][-1]
#     print(acc)

