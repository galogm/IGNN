import argparse
import random
import sys
import time

sys.path.append('../../..')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from dataset import load_dataset
from eval import *
from logger import *
from parse import parse_method, parser_add_main_args
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)

# def fix_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

### Parse args ###
parser = argparse.ArgumentParser(
    description="Training Pipeline for Node Classification"
)
parser_add_main_args(parser)
args = parser.parse_args()
if not args.global_dropout:
    args.global_dropout = args.dropout
print(args)

# fix_seed(args.seed)

# if args.cpu:
#     device = torch.device("cpu")
# else:
#     device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
# dataset = load_dataset(args.data_dir, args.dataset)

from the_utils import (save_to_csv_files, set_device, set_seed,
                       split_train_test_nodes)

set_seed(args.seed)
device = set_device(args.device)
from graph_datasets import load_data

from ignn.modules import DataConf
from ignn.utils import read_configs

DATA = DataConf(**read_configs("data"))

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
dataset.edge_index = dataset.edge_index.to(torch.int64)


# if len(dataset.label.shape) == 1:
#     dataset.label = dataset.label.unsqueeze(1)
# dataset.label = dataset.label.to(device)


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


# split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

### Basic information of datasets ###
# n = dataset.graph['num_nodes']
# e = dataset.graph['edge_index'].shape[1]
# c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
# d = dataset.graph['node_feat'].shape[1]
n = dataset.num_nodes
e = dataset.edge_index.shape[1]
c = dataset.num_classes
d = dataset.feat.shape[1]

# print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

# dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
# dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
# dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

# dataset.graph['edge_index'], dataset.graph['node_feat'] = \
#     dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)


### Loss function (Single-class, Multi-class) ###
if args.dataset in ("questions"):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == "rocauc":
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

### Training loop ###
t_s = time.time()
for run in range(args.runs):
    ### Load method ###
    model = parse_method(args, n, c, d, device)
    model.train()
    print("MODEL:", model)
    # if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo'):
    #     split_idx = split_idx_lst[0]
    # else:
    #     split_idx = split_idx_lst[run]
    split_idx = get_splits_mask(
        n,
        dataset.name,
        48,
        32,
        args.runs,
        run,
        DATA.SPLIT_DIR,
        mask=False,
    )

    train_idx = split_idx["train"].to(device)
    model.reset_parameters()
    model._global = False
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay, lr=args.lr
    )
    best_val = float("-inf")
    best_test = float("-inf")

    if not os.path.exists(f'.checkpoints/models/{args.dataset}'):
        os.makedirs(f'.checkpoints/models/{args.dataset}')
    model_path = f'.checkpoints/models/{args.dataset}/{args.method}_{run}_{args.beta}-{time.strftime("%Y-%m-%d_%H-%M-%S")}-{random.Random().uniform(1,1e4):.0f}-.pt'

    # if args.save_model:
    #     save_model(args, model, optimizer, run, model_path)

    for epoch in range(args.local_epochs + args.global_epochs):
        if epoch == args.local_epochs:
            print("start global attention!!!!!!")
            # if args.save_model:
            #     model, optimizer = load_model(args, model, optimizer, run, model_path)
            model._global = True
        model.train()
        optimizer.zero_grad()

        out = model(dataset.feat, dataset.edge_index)
        if args.dataset in ("questions"):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1)
            else:
                true_label = dataset.label
            loss = criterion(
                out[train_idx], true_label[train_idx].to(torch.float)
            )
        else:
            out = F.log_softmax(out, dim=1)
            # print(train_idx, dataset.label.squeeze(1))
            loss = criterion(out[train_idx], dataset.label[train_idx])
        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            cnt = 0
            best_val = result[1]
            best_test = result[2]
            # if args.save_model:
            #     save_model(args, model, optimizer, run,model_path)
        else:
            cnt = cnt + 1

        if epoch % args.display_step == 0:
            print(
                f"Epoch: {epoch:02d}, "
                f"Loss: {loss:.4f}, "
                f"Train: {100 * result[0]:.2f}%, "
                f"Valid: {100 * result[1]:.2f}%, "
                f"Test: {100 * result[2]:.2f}%, "
                f"Best Valid: {100 * best_val:.2f}%, "
                f"Best Test: {100 * best_test:.2f}%"
            )

        if cnt == args.estop_steps:
            break
    logger.print_statistics(run)

time_elapsed = (time.time()-t_s)/args.runs

results = logger.print_statistics()
### Save results ###
# save_result(args, results)

save_to_csv_files(
    results=results,
    insert_info={
        "dataset": dataset.name,
        "model": "Polynormer",
    },
    append_info={
        "time": f"{time_elapsed:.2f}",
        "args": args.__dict__,
        "source": args.source,
    },
    csv_name=f"baselines.csv",
)
