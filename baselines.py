# pylint: disable=unused-import
import argparse
import os
import random
import sys
import time
from copy import deepcopy

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from graph_datasets import load_data
from sklearn.metrics import accuracy_score as ACC
from the_utils import save_to_csv_files, set_device, set_seed, tab_printer

import benchmark.baselines as baselines
from benchmark.modules import Data
from benchmark.utils import get_splits_mask, set_args_wrt_dataset
from ignn.configs import DataConf
from utils import read_configs

DATASETS = {
    "critical": [
        # 890
        "chameleon",
        # 2,223
        "squirrel",
        # # 10,000
        # "minesweeper",
        # # 11,758
        # "tolokers",
        # # 22,662
        # "roman-empire",
        # # 24,492
        # "amazon-ratings",
        # 48,921
        # "questions",
    ],
    "cola": [
        "flickr",
        "blogcatalog",
    ],
    "pyg": [
        # "texas",
        # "cornell",
        # "wisconsin",
        #     (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        #    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        #    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        #    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
        #    68, 69]), array([257,  52, 243, 378,  63, 305, 404, 663, 240, 342, 141, 223, 102,
        #    521, 341, 138, 115, 111,  80, 435, 420, 254, 414, 196, 334, 315,
        #    284, 783, 113, 466, 221, 376, 154, 855, 576,  84, 293, 163, 125,
        #    564, 280, 205,  94,  53, 129, 370, 122,  74, 557, 285,  72, 625,
        #    501, 650,  99, 473, 324, 928, 212, 301, 116, 220, 165, 291, 147,
        #     91, 137,  84,  15,  29]))
        # "corafull",
        # "cora",
        # "citeseer",
        "photo",
        "actor",
        "pubmed",
        "wikics",
    ],
    "ogb": [
        "arxiv",
    ],
    "Critical": [
        # 22,662
        "roman-empire",
        # 24,492
        "amazon-ratings",
        # 48,921
        # "questions",
    ],
    # "linkx": [
    #     # (array([0, 1, 2]), array([ 97, 504, 361]))
    #     # 962
    #     "Reed98",
    #     # array([0, 1, 2]), array([ 418, 2153, 2609]
    #     # 5,180
    #     "Johns Hopkins55",
    #     # (array([0, 1, 2]), array([ 203, 1015, 1017]))
    #     # 2,235
    #     "Amherst41",
    #     # (array([0, 1, 2]), array([1838, 8135, 8687]))
    #     # 18,660
    #     "Cornell5",
    #     # 41,554
    #     "Penn94",
    #     # 168,114
    #     "twitch-gamers",
    #     # 169,343
    #     "arxiv-year",
    #     # 421,961
    #     "genius",
    #     # 1,632,803
    #     "pokec",
    #     # 2,923,922
    #     "snap-patents",
    # ],
}

model_keys = {
    "GCN": "layers",
    "SGC": "k",
    "GAT": "layers",
    "GraphSAGE": "layers",
    "APPNP": "iterations",
    "SIGN": "n_hops",
    "IncepGCN": "n_layers",
    "JKNet": "n_layers",
    "MixHop": None,
    "DAGNN": "k",
    "GCNII": "layers",
    "H2GCN": "k",
    "GBKGNN": None,
    "GGNN": "nlayers",
    "GloGNN": "orders",
    "HOGGCN": None,
    "GPRGNN": "nlayers",
    "ACMGCN": None,
    "OrderedGNN": "num_layers",
    "NodeFormer": "num_layers",
    "DIFFormer": "num_layers",
    "SGFormer": "num_layers",
    "IGNN": "n_hops",
}


def main(dataset, source):
    DEVICE = set_device(args.gpu)
    DATA = DataConf(**read_configs("data"))
    graph, label, class_num = load_data(
        dataset_name=dataset,
        directory=DATA.DATA_DIR,
        source=source,
        row_normalize=True,
        rm_self_loop=False,
        add_self_loop=True,
        to_simple=True,
        verbosity=3,
    )
    if args.model in ["GloGNN", "ACMGCN"]:
        args.nnodes = graph.num_nodes()
    elif args.model == "HOGGCN":
        adj = graph.adj_external(scipy_fmt="csr")
        args.adj = torch.tensor(adj.todense(), dtype=torch.float32)
    elif args.model == "APPNP":
        args.adj_csr = graph.remove_self_loop().adj_external(scipy_fmt="csr")

    t_start = time.time()
    set_seed(args.seed)
    seed_list = [random.randint(0, 99999) for i in range(args.runs * 100)]

    res_list_acc = []
    if model_keys[args.model] is not None:
        setattr(args, model_keys[args.model], args.n_hops)
    tab_printer(args)
    tms = {
        "model": args.model,
        "dataset": args.dataset,
        "hops": (
            getattr(args, model_keys[args.model])
            if model_keys[args.model]
            else "architecturally limited"
        ),
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "mean": 0,
    }
    ts = []
    for run in range(args.runs):
        print(f"\nRun: {run}\n")
        set_seed(seed_list[run])
        if graph.name == "arxiv_ogb":
            train_mask, val_mask, test_mask = (
                graph.ndata["train_mask"],
                graph.ndata["val_mask"],
                graph.ndata["test_mask"],
            )
        else:
            train_mask, val_mask, test_mask = get_splits_mask(
                graph=graph,
                train_ratio=48,
                valid_ratio=32,
                repeat=args.runs,
                split_id=run,
                SPLIT_DIR=DATA.SPLIT_DIR,
            )

        Model = getattr(
            baselines,
            (
                args.model
                if args.model != "JKNet"
                else {
                    "concat": "JKNetConcat",
                    "max": "JKNetMaxpool",
                }[args.type]
            ),
        )
        model = Model(
            in_features=graph.ndata["feat"].shape[1],
            class_num=class_num,
            device=DEVICE,
            args=args,
        )
        t_m = model.fit(
            graph,
            label,
            train_mask,
            val_mask,
            test_mask,
        )
        tms[f"{run}"] = t_m
        ts.append(t_m)
        res, C, Z = model.predict(graph)
        acc = ACC(label[test_mask], res[test_mask])
        res_list_acc.append(acc)
        print(f"Acc: {acc}")

    save_to_csv_files(
        results={**tms},
        append_info={
            "mean": f"{np.array(ts).mean():.2f}±{np.array(ts).std():.2f}",
        },
        csv_name="times.csv",
    )
    elapsed_time = f"{time.time() - t_start:.2f}"
    print(f"Train cost: {elapsed_time}s")
    acc_ = f"{np.array(res_list_acc).mean() * 100:.2f}±{np.array(res_list_acc).std() * 100:.2f}"
    print(f"\nResults:\t{acc_}")
    save_to_csv_files(
        results={
            "acc": acc_,
            "time": elapsed_time,
        },
        insert_info={
            "dataset": dataset,
            "model": model._get_name(),
        },
        append_info={
            "args": args.__dict__,
            "source": source,
        },
        csv_name="baselines.csv",
    )


def set_args(args):
    if args.model == "GloGNN":
        # args.nhid = 64
        args.nhid = 512
        args.orders = args.n_hops
        args.orders_func_id = 2
    elif args.model == "GCNII":
        # args.nhidden = 64
        args.nhidden = 512
        args.layers = args.n_hops
    elif args.model == "GGCN":
        args.use_degree = True
        args.use_sign = True
        args.use_decay = True
        args.use_bn = False
        args.use_ln = False
        args.exponent = 3.0
        args.scale_init = 0.5
        args.deg_intercept_init = 0.5
    elif args.model == "ACMGCN":
        # args.nhid = 64
        args.nhid = 256
        args.nlayers = 1
        args.init_layers_X = 1
        args.acm_model = "acmgcnp"
    elif args.model == "OrderedGNN":
        args.chunk_size = 64
        args.hidden_channel = 512
        args.num_layers = args.n_hops
        args.add_self_loops = False
        args.simple_gating = False
        args.tm = True
        args.diff_or = True
    elif args.model == "GPRGNN":
        args.init = "PPR"
        args.dropout = 0.5
        # args.nhidden = 64
        args.nhidden = 256
        args.gamma = None
    elif args.model == "GBKGNN":
        # args.dim_size = 16
        args.dim_size = 256
    elif args.model == "HOGGCN":
        args.nhid1 = 256
        args.nhid2 = 256
        args.dropout = 0.5
    elif args.model == "H2GCN":
        args.k = 1
        # args.hidden_dim = 512
        args.hidden_dim = 256
    elif args.model == "MixHop":
        args.layers_1 = [200, 200, 200]
        args.dropout = 0.5
        args.layers_2 = [200, 200, 200]
        args.lambd = 0.0005
        args.cut_off = 0.1
        args.budget = 60
    elif args.model == "APPNP":
        # args.layers = [512, 512]
        args.layers = [256, 256]
    elif args.model == "IncepGCN":
        args.n_layers = args.n_hops
    elif args.model == "DAGNN":
        args.k = args.n_hops
    elif args.model in [
        "MLP",
        "GCN",
        "GAT",
        "GraphSAGE",
        "JKNet",
        "SIGN",
        "IncepGCN",
        "SGC",
        "DAGNN",
        "NodeFormer",
        "DIFFormer",
        "SGFormer",
    ]:
        pass
    elif args.model == "CMGNN":
        args.self_loop = False
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="All", description="Parameters for Baseline Method")
    # experiment parameter
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Defaults to 4096.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument(
        "--runs", type=int, default=10, help="The number of runs of task with same parmeter"
    )
    parser.add_argument(
        "--dataset", type=str, default="chameleon", help="Dataset used in the experiment"
    )
    parser.add_argument(
        "--source", type=str, default="critical", help="Dataset source used in the experiment"
    )

    # train parameter
    parser.add_argument("--epochs", type=int, default=2000, help="num of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--l2_coef", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=200, help="early stop patience")

    # model parameter
    parser.add_argument("--nhidden", type=int, default=128, help="num of hidden dimension")
    parser.add_argument("--undirected", type=bool, default=True, help="change graph to undirected")
    parser.add_argument("--self_loop", type=bool, default=True, help="add self loop")
    parser.add_argument(
        "--normalize", type=int, default=-1, help="feature norm, -1 for without normalize"
    )
    parser.add_argument("--layers", type=int, default=1, help="layers of model")
    parser.add_argument("--n_hops", type=int, default=1, help="hops")
    parser.add_argument("--model", type=str, default="GCN", help="")

    args = parser.parse_args()
    n_hops = args.n_hops

    DIM_BOUND = {
        "pubmed": 500,
        "roman-empire": 300,
        "amazon-ratings": 300,
        "wikics": 300,
    }

    if args.dataset != "all":
        args_dict = args.__dict__
        args = set_args_wrt_dataset(args, args_dict)
        args.model = args_dict["model"]
        set_args(args)
        args.n_hops = n_hops
        main(
            dataset=args.dataset,
            source=args.source,
        )
    else:
        for source, datasets in DATASETS.items():
            for dataset in datasets:
                try:
                    source = source.lower()
                    args.dataset = dataset
                    args.source = source
                    args_dict = args.__dict__
                    args = set_args_wrt_dataset(args, args_dict)
                    args.model = args_dict["model"]
                    set_args(args)
                    main(
                        dataset=dataset,
                        source=source,
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    continue
