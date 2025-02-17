"""Full Batch"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from graph_datasets import load_data
from the_utils import draw_chart
from the_utils import set_device
from the_utils import set_seed
from torch import nn

from ignn.utils import eval_rocauc

torch.set_printoptions(threshold=10_000)
np.set_printoptions(threshold=10_000)

from ignn.models import IGNN
from ignn.modules import Data
from ignn.utils import get_splits_mask
from ignn.utils import read_configs
from ignn.utils import set_args_wrt_dataset, metric

from the_utils import (
    evaluate_from_embeddings,
    tab_printer,
    save_to_csv_files,
    split_train_test_nodes,
    check_modelfile_exists,
    make_parent_dirs,
)

import networkx as nx
import time

from sklearn.metrics import accuracy_score as ACC
import traceback

norms = {
    "chameleon": False,
    "squirrel": True,
    "actor": False,
    "flickr": True,
    "blogcatalog": True,
    "roman-empire": False,
    "amazon-ratings": True,
    "photo": True,
    "pubmed": True,
    "wikics": False,
    "arxiv": False,
    "products": False,
    "pokec": False,
}

feats = {
    "chameleon": 512,
    "squirrel": 512,
    "actor": 512,
    "photo": 256,
    "pubmed": 500,
    "roman-empire": 300,
    "amazon-ratings": 300,
    "flickr": 512,
    "blogcatalog": 512,
    "wikics": 300,
    "arxiv": 512,
    "products": 512,
    "pokec": 512,
}

l2_coefs = {
    "chameleon": 0.00005,
    "squirrel": 0.00005,
    "actor": 0.0000,
    "pubmed": 0.00005,
    "photo": 0.00005,
    "roman-empire": 0.00005,
    "amazon-ratings": 0.00005,
    "flickr": 0.00005,
    "blogcatalog": 0.00005,
    "wikics": 0.00005,
    "arxiv": 0.00005,
    "products": 0.00005,
    "pokec": 0.00005,
}
bss = {
    "chameleon": 100000,
    "squirrel": 100000,
    "actor": 100000,
    "pubmed": 100000,
    "photo": 100000,
    "roman-empire": 100000,
    "amazon-ratings": 100000,
    "flickr": 100000,
    "blogcatalog": 100000,
    "wikics": 100000,
    "arxiv": 100000,
    "products": 100000,
    "pokec": 100000,
}
lrs = {
    "chameleon": 0.001,
    "squirrel": 0.001,
    "actor": 0.001,
    "pubmed": 0.001,
    "photo": 0.001,
    "roman-empire": 0.001,
    "amazon-ratings": 0.001,
    "flickr": 0.001,
    "blogcatalog": 0.001,
    "wikics": 0.001,
    "arxiv": 0.001,
    "products": 0.001,
    "pokec": 0.001,
}
ess = {
    "chameleon": 30,
    "squirrel": 50,
    "actor": 200,
    "pubmed": 100,
    "photo": 100,
    "roman-empire": 100,
    "amazon-ratings": 100,
    "flickr": 100,
    "blogcatalog": 100,
    "wikics": 200,
    "arxiv": 200,
    "products": 200,
    "pokec": 200,
}
nas_dropouts = {
    "flickr": 0.8,
    "blogcatalog": 0.8,
    "chameleon": 0.8,
    "squirrel": 0.8,
    "actor": 0.0,
    "pubmed": 0.2,
    "photo": 0.4,
    "roman-empire": 0.2,
    "amazon-ratings": 0.0,
    "wikics": 0.2,
    "arxiv": 0.0,
    "products": 0.0,
    "pokec": 0.0,
}
nss_dropouts = {
    "flickr": 0.8,
    "blogcatalog": 0.8,
    "chameleon": 0.8,
    "squirrel": 0.8,
    "actor": 0.8,
    "pubmed": 0.8,
    "photo": 0.5,
    "roman-empire": 0.2,
    "amazon-ratings": 0.5,
    "wikics": 0.5,
    "arxiv": 0.8,
    "products": 0.8,
    "pokec": 0.8,
}
# higher n_feats, higher clf_dropout
clf_dropouts = {
    "flickr": 0.9,
    "blogcatalog": 0.9,
    "chameleon": 0.9,
    "squirrel": 0.9,
    "actor": 0.9,
    "photo": 0.5,
    "pubmed": 0.9,
    "roman-empire": 0.2,
    "amazon-ratings": 0.9,
    "wikics": 0.9,
    "arxiv": 0.5,
    "products": 0.5,
    "pokec": 0.5,
}
n_hopss = {
    "chameleon": 64,
    "squirrel": 64,
    "actor": 1,
    "flickr": 10,
    "blogcatalog": 10,
    "roman-empire": 1,
    "amazon-ratings": 10,
    "photo": 10,
    "pubmed": 4,
    "wikics": 8,
    "arxiv": 10,
    "products": 10,
    "pokec": 10,
}
n_layers = {
    "chameleon": 1,
    "squirrel": 1,
    "actor": 1,
    "flickr": 1,
    "blogcatalog": 1,
    "roman-empire": 3,
    "amazon-ratings": 1,
    "photo": 1,
    "pubmed": 1,
    "wikics": 1,
    "arxiv": 1,
    "products": 1,
    "pokec": 1,
}
nrls = {
    "chameleon": "concat",
    "squirrel": "concat",
    "actor": "concat",
    "flickr": "concat",
    "blogcatalog": "concat",
    "roman-empire": "concat",
    "amazon-ratings": "concat",
    "pubmed": "concat",
    "photo": "concat",
    "wikics": "concat",
    "arxiv": "concat",
    "products": "concat",
    "pokec": "concat",
}
acts = {
    "chameleon": "relu",
    "squirrel": "relu",
    "actor": "relu",
    "flickr": "relu",
    "blogcatalog": "relu",
    "roman-empire": "relu",
    "amazon-ratings": "prelu",
    "photo": "relu",
    "pubmed": "relu",
    "wikics": "relu",
    "arxiv": "relu",
    "products": "relu",
    "pokec": "relu",
}
layer_norms = {
    "chameleon": True,
    "squirrel": True,
    "actor": True,
    "flickr": True,
    "blogcatalog": True,
    "roman-empire": True,
    "amazon-ratings": True,
    "photo": True,
    "pubmed": True,
    "wikics": True,
    "arxiv": True,
    "products": True,
    "pokec": True,
}

layer_norms_att = {
    "chameleon": False,
    "squirrel": False,
    "actor": True,
    "flickr": False,
    "blogcatalog": True,
    "roman-empire": False,
    "amazon-ratings": True,
    "photo": True,
    "pubmed": True,
    "wikics": False,
    "arxiv": False,
    "products": False,
    "pokec": False,
}

n_intervalss = {
    "chameleon": 3,
    "squirrel": 3,
    "actor": 3,
    "flickr": 3,
    "blogcatalog": 3,
    "roman-empire": 3,
    "amazon-ratings": 3,
    "photo": 3,
    "pubmed": 3,
    "wikics": 3,
    "arxiv": 3,
    "products": 3,
    "pokec": 3,
}
self_loop_attentive = {
    "chameleon": False,
    "squirrel": False,
    "actor": False,
    "flickr": False,
    "blogcatalog": False,
    "roman-empire": False,
    "amazon-ratings": False,
    "photo": False,
    "pubmed": False,
    "wikics": False,
    "arxiv": False,
    "products": False,
    "pokec": False,
}

# TRAIN_RATIO = 10
# VALID_RATIO = 10
# TRAIN_RATIO = 50
# VALID_RATIO = 25


def graph_diameter(g):
    """
    Computes the diameter of a given graph.

    Parameters:
    g (DGLGraph): The input graph.

    Returns:
    int: The diameter of the graph.
    """
    # Convert the DGL graph to a NetworkX graph
    nx_graph = g.to_networkx()

    # Use NetworkX to compute the all-pairs shortest path lengths
    lengths = dict(nx.all_pairs_shortest_path_length(nx_graph))

    # Find the maximum shortest path length
    max_length = 0
    for src in lengths:
        for dst in lengths[src]:
            if lengths[src][dst] > max_length:
                max_length = lengths[src][dst]

    return max_length


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
        # "corafull",
        # "cora",
        # "citeseer",
        "photo",
        "actor",
        "pubmed",
        "wikics",
    ],
    "Critical": [
        # 22,662
        "roman-empire",
        # 24,492
        "amazon-ratings",
        # 48,921
        # "questions",
    ],
    "ogb": [
        "arxiv",
        "proteins",
    ],
    "linkx": [
        # (array([0, 1, 2]), array([ 97, 504, 361]))
        # 962
        "Reed98",
        # # array([0, 1, 2]), array([ 418, 2153, 2609]
        # # 5,180
        "Johns Hopkins55",
        # # (array([0, 1, 2]), array([ 203, 1015, 1017]))
        # # 2,235
        "Amherst41",
        # # (array([0, 1, 2]), array([1838, 8135, 8687]))
        # # 18,660
        "Cornell5",
        # # 41,554
        "Penn94",
        # # 168,114
        "twitch-gamers",
        # 169,343
        "arxiv-year",
        # 421,961
        "genius",
        # # 1,632,803
        "pokec",
        # 2,923,922
        "snap-patents",
        # 1,925,342
        "wiki",
    ],
}


def main(
    dataset,
    source,
    h_feats,
    MODEL,
    BATCH_SIZE=None,
    nie=None,
    nrl=None,
    n_hops=None,
    n_layers=None,
    lr=None,
    l2_coefs=None,
    nas_dropout=None,
    nss_dropout=None,
    clf_dropout=None,
):
    BATCH_SIZE = BATCH_SIZE
    graph, label, n_clusters = load_data(
        dataset_name=dataset,
        directory=DATA.DATA_DIR,
        source=source,
        row_normalize=norms[dataset],
        rm_self_loop=False if nrl != "attentive" else (not self_loop_attentive[dataset]),
        # rm_self_loop=False,
        # add_self_loop=False if nrl not in ["residual", "attentive"] else True,
        add_self_loop=True if nrl != "attentive" else self_loop_attentive[dataset],
        # products and arxiv-year are already simple graphs
        to_simple=True if dataset not in ["products", "arxiv-year"] else False,
        verbosity=3,
    )

    repeat = (
        3
        if dataset
        in [
            "pokec",
            "arxiv",
            "products",
            "proteins",
            "wiki",
            "twitch-gamers",
            "snap-patents",
            "arxiv-year",
            "genius",
            "Penn94",
        ]
        else 10
    )

    print(label.min(), label.max())
    if dataset in ["wiki", "Penn94", "pokec"]:
        labeled_idx = torch.where(label != -1)[0]
        n_clusters = 2 if dataset == "pokec" else n_clusters
        # label=labeled_nodes
    # if dataset=="pokec":
    #     labeled_idx = torch.where(label != 2)[0]
    #     if label.max().item() == 2:
    #         label[label == 2] = 0

    # save_to_csv_files(
    #     results={
    #         "diameter": graph_diameter(graph),
    #     },
    #     insert_info={
    #         "dataset": dataset,
    #     },
    #     append_info={
    #         "bs": BATCH_SIZE,
    #         "source": source,
    #     },
    #     csv_name=f"diameter.csv",
    # )

    TRAIN_RATIO = 48
    VALID_RATIO = 32
    # if dataset in ["wiki"]:
    #     TRAIN_RATIO = 10
    #     VALID_RATIO = 10
    # elif dataset in [
    #     "twitch-gamers",
    #     "snap-patents",
    #     "arxiv-year",
    #     "Penn94",
    #     "genius",
    #     "pokec",
    # ]:
    #     TRAIN_RATIO = 50
    #     VALID_RATIO = 25
    # elif dataset in ["cora", "pubmed"]:
    #     TRAIN_RATIO = 60
    #     VALID_RATIO = 20

    N_HOPS = n_hops if n_hops is not None else n_hopss[dataset]
    N_LAYERS = (
        n_layers
        if n_layers is not None
        else n_layers[dataset] if nrl not in ["residual", "attentive"] else 1
    )
    LR = lr if lr is not None else lrs[dataset]
    COEF = l2_coefs if l2_coefs is not None else l2_coefs[dataset]
    DNAS = nas_dropout if nas_dropout is not None else nas_dropouts[dataset]
    DNSS = nss_dropout if nss_dropout is not None else nss_dropouts[dataset]
    DCLF = clf_dropout if clf_dropout is not None else clf_dropouts[dataset]
    params = {
        "nie": nie,
        "nrl": nrl if nrl is not None else nrls[dataset],
        "lr": LR,
        "h_feats": min(h_feats, feats[dataset]),
        "l2_coef": COEF,
        "nas_dropout": DNAS,
        "nss_dropout": DNSS,
        "clf_dropout": DCLF,
        "n_epochs": 2000,
        "n_hops": N_HOPS,
        "n_layers": N_LAYERS,
        "early_stop": ess[dataset],
        "n_intervals": min(n_intervalss[dataset], N_HOPS + 1),
        "act": acts[dataset],
        "layer_norm": layer_norms[dataset] if nrl != "attentive" else layer_norms_att[dataset],
        "loss": "ce" if dataset not in ["proteins"] else "bce",
        "n_nodes": (
            graph.num_nodes()
            if dataset
            not in [
                "genius",
                "cora",
                "actor",
                "arxiv",
                "pokec",
                "Penn94",
                "snap-patents",
                "products",
            ]
            and graph.num_nodes() >= 4e4
            else None
        ),
        "transform_first": False,
    }
    params_all = {
        "row_normalized": norms[dataset],
        **params,
    }
    tab_printer({"bs": BATCH_SIZE, **params_all})

    t_start = time.time()

    seed_list = [random.randint(0, 99999) for i in range(repeat)]

    res_list_acc_joint = []
    print("train_mask exists:", "train_mask" in graph.ndata)

    tms = {
        "model": "IGNN",
        "hops": N_HOPS,
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
    for i in range(repeat):
        graph.split = i
        # set_seed(seed_list[i])
        # model.load_state_dict(torch.load(STATE, map_location=DEVICE))
        model = IGNN(
            in_feats=graph.ndata["feat"].shape[1],
            n_clusters=n_clusters if dataset not in ["proteins"] else label.shape[1],
            device=DEVICE,
            **params,
        )

        if "train_mask" in graph.ndata and graph.name not in [
            "cora_pyg",
            "actor_pyg",
            "pubmed_pyg",
            "wikics_pyg",
            "arxiv-year_linkx",
        ]:
            train_mask, val_mask, test_mask = (
                graph.ndata["train_mask"],
                graph.ndata["val_mask"],
                graph.ndata["test_mask"],
            )
            split_ratio = (
                np.array(
                    [
                        graph.ndata["train_mask"].sum(),
                        graph.ndata["val_mask"].sum(),
                        graph.ndata["test_mask"].sum(),
                    ]
                )
                / graph.num_nodes()
            )
            print(
                f"public split train:val:test = {split_ratio[0]*100:.0f}:{split_ratio[1]*100:.0f}:{split_ratio[2]*100:.0f}"
            )
        elif graph.name == "pokec_linkx":
            # splits_lst = []
            split = np.load(
                f"data/random_splits/fixed_splits/pokec-splits.npy", allow_pickle=True
            )
            print("split num:", split.shape)
            train_idx = torch.from_numpy(np.asarray(split[i]["train"]))
            val_idx = torch.from_numpy(np.asarray(split[i]["valid"]))
            test_idx = torch.from_numpy(np.asarray(split[i]["test"]))

            train_idx = train_idx[label[train_idx] != -1]
            val_idx = val_idx[label[val_idx] != -1]
            test_idx = test_idx[label[test_idx] != -1]

            # train_idx = labeled_idx[train_idx]
            # val_idx = labeled_idx[val_idx]
            # test_idx = labeled_idx[test_idx]

            train_mask = (
                torch.zeros(graph.num_nodes())
                .scatter_(
                    0,
                    (
                        train_idx.to(torch.int64)
                        if isinstance(train_idx, torch.Tensor)
                        else torch.LongTensor(train_idx)
                    ),
                    1,
                )
                .bool()
            )
            val_mask = (
                torch.zeros(graph.num_nodes())
                .scatter_(
                    0,
                    (
                        val_idx.to(torch.int64)
                        if isinstance(val_idx, torch.Tensor)
                        else torch.LongTensor(val_idx)
                    ),
                    1,
                )
                .bool()
            )
            test_mask = (
                torch.zeros(graph.num_nodes())
                .scatter_(
                    0,
                    (
                        test_idx.to(torch.int64)
                        if isinstance(test_idx, torch.Tensor)
                        else torch.LongTensor(test_idx)
                    ),
                    1,
                )
                .bool()
            )
            split_ratio = (
                np.array(
                    [
                        train_mask.sum().item(),
                        val_mask.sum().item(),
                        test_mask.sum().item(),
                    ]
                )
                / graph.num_nodes()
            )
            print(
                f"public split train:val:test = {split_ratio[0]*100:.0f}:{split_ratio[1]*100:.0f}:{split_ratio[2]*100:.0f}"
            )
        else:
            train_mask, val_mask, test_mask = get_splits_mask(
                name=graph.name,
                n_nodes=graph.num_nodes(),
                train_ratio=TRAIN_RATIO,
                valid_ratio=VALID_RATIO,
                repeat=10,
                split_id=i,
                SPLIT_DIR=DATA.SPLIT_DIR,
                labeled_idx=(
                    labeled_idx if graph.name in ["wiki_linkx", "pokec_linkx"] else None
                ),
            )
            print(
                f"random split train:val:test = {TRAIN_RATIO}:{VALID_RATIO}:{100-TRAIN_RATIO-VALID_RATIO}"
            )

        # print(label[train_mask].min(), label[val_mask].min(), label[test_mask].min())
        tm = model.fit(
            graph=graph,
            labels=label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            split_id=i,
            bs=BATCH_SIZE,
            device=DEVICE,
        )

        if BATCH_SIZE is not None:
            with torch.no_grad():
                model.train(False)
                pred_stack = []
                idx = torch.LongTensor(list(range(graph.num_nodes())))
                for _, step in enumerate(range(0, graph.num_nodes(), BATCH_SIZE)):
                    batch_idx = idx[step : step + BATCH_SIZE]

                    embeddings = model.forward(
                        graph.ndata["feat"],
                        batch_idx=batch_idx,
                        graph=graph,
                        device=DEVICE,
                    )
                    # H,embeddings = model.forward(
                    #     graph.ndata["feat"],
                    #     batch_idx=batch_idx,
                    #     graph=graph,
                    #     device=DEVICE,
                    # )
                    logits = model.classifier(embeddings)
                    pred_stack.append(logits)

                y_pred = torch.cat(pred_stack, dim=0)

                train_acc, val_acc, res = metric(
                    graph.name,
                    logits=y_pred,
                    labels=label,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                )

                # if graph.name not in (
                #     "yelp-chi",
                #     "deezer-europe",
                #     "twitch-gamers",
                #     "pokec",
                #     "Penn94_linkx",
                #     "fb100",
                #     "proteins_ogb",
                #     "pokec_linkx",
                # ):
                #     y_pred = torch.argmax(y_pred, dim=1)
                #     res = ACC(label[test_mask].cpu(), y_pred[test_mask].cpu())
                # else:
                #     if graph.name in [
                #         "proteins_ogb",
                #         # "Penn94_linkx",
                #         # "pokec_linkx",
                #     ]:
                #         res = eval_rocauc(
                #             label[test_mask].cpu().numpy(), y_pred[test_mask].cpu().numpy()
                #         )["rocauc"]
        else:
            with torch.no_grad():
                model.train(False)
                embeddings = model(
                    graph.ndata["feat"],
                    graph=graph,
                    device=DEVICE,
                )
                # H,embeddings = model(
                #     graph.ndata["feat"],
                #     graph=graph,
                #     device=DEVICE,
                # )
                y_pred = model.classifier(embeddings)
                # y_pred = torch.argmax(logits_onehot, dim=1)

                train_acc, val_acc, res = metric(
                    graph.name,
                    logits=y_pred,
                    labels=label,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                )

                # if graph.name not in (
                #     "yelp-chi",
                #     "deezer-europe",
                #     "twitch-gamers",
                #     "pokec",
                #     "Penn94_linkx",
                #     "fb100",
                #     "proteins_ogb",
                #     "pokec_linkx",
                # ):
                #     print(f'Evaluate {graph.name} with accuracy.')
                #     y_pred = torch.argmax(y_pred, dim=1)
                #     res = ACC(label[test_mask].cpu(), y_pred[test_mask].cpu())
                # else:
                #     if graph.name in [
                #         "proteins_ogb",
                #         "genius_linkx"
                #         # "Penn94_linkx",
                #         # "pokec_linkx",
                #     ]:
                #         print(f'Evaluate {graph.name} with rocauc.')
                #         res = eval_rocauc(
                #             label[test_mask].cpu().numpy(), y_pred[test_mask].cpu().numpy()
                #         )["rocauc"]
        tms[f"{i}"] = tm
        ts.append(tm)

        res_list_acc_joint.append(res)
        print(f"{graph.name} {i} res: {res}\n\n")

    save_to_csv_files(
        results={**tms},
        append_info={
            "mean": f"{np.array(ts).mean():.2f}±{np.array(ts).std():.2f}",
        },
        csv_name="times.csv",
    )
    elapsed_time = f"{(time.time() - t_start)/repeat:.2f}"
    acc_jl = f"{np.array(res_list_acc_joint).mean() * 100:.2f}±{np.array(res_list_acc_joint).std() * 100:.2f}"

    print(f"\nResults: \tAcc:{acc_jl} \tTrain cost: {elapsed_time}s")
    save_to_csv_files(
        results={
            "acc_hl": acc_jl,
            "hop": N_HOPS,
        },
        insert_info={
            "dataset": dataset,
        },
        append_info={
            "args": params_all,
            "time": elapsed_time,
            "source": source,
            "model": MODEL,
            "bs": BATCH_SIZE,
        },
        csv_name=f"results_v{VERSION}.csv",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="IGNN",
        description="",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=int,
        default=0,
        help="gpu id",
    )
    parser.add_argument(
        "-f",
        "--h_feats",
        type=int,
        default=512,
        help="h feats",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="dataset",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default=None,
        help="source",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gcn-nie-nst",
        help="model",
    )
    parser.add_argument(
        "-nie",
        "--nie",
        type=str,
        default="gcn-nie-nst",
        help="nie",
    )
    parser.add_argument(
        "-nrl",
        "--nrl",
        type=lambda x: None if x == "None" else x,
        default=None,
        help="nrl",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=float,
        default=1,
        help="version",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=None,
        help="batch size",
    )
    parser.add_argument(
        "-hops",
        "--n_hops",
        type=lambda x: None if x == "None" else int(x),
        default=None,
        help="n_hops",
    )
    parser.add_argument(
        "-layers",
        "--n_layers",
        type=lambda x: None if x == "None" else int(x),
        default=None,
        help="n_layers",
    )
    parser.add_argument(
        "-lr",
        "--lr",
        type=lambda x: None if x == "None" else float(x),
        default=None,
        help="lr",
    )
    parser.add_argument(
        "-l2_coefs",
        "--l2_coefs",
        type=lambda x: None if x == "None" else float(x),
        default=None,
        help="l2_coefs",
    )
    parser.add_argument(
        "-nas_dropout",
        "--nas_dropout",
        type=lambda x: None if x == "None" else float(x),
        default=None,
        help="nas_dropout",
    )
    parser.add_argument(
        "-nss_dropout",
        "--nss_dropout",
        type=lambda x: None if x == "None" else float(x),
        default=None,
        help="nss_dropout",
    )
    parser.add_argument(
        "-clf_dropout",
        "--clf_dropout",
        type=lambda x: None if x == "None" else float(x),
        default=None,
        help="clf_dropout",
    )
    args = parser.parse_args()

    DATA = Data(**read_configs("data"))
    VERSION = args.version

    DEVICE = set_device(str(args.gpu_id))
    set_seed(42)

    args.batch_size = args.batch_size or None

    DIM_BOUND = {
        "pubmed": 500,
        "photo": 256,
        "roman-empire": 300,
        "amazon-ratings": 300,
        "wikics": 300,
    }

    if args.dataset != "all":
        main(
            dataset=args.dataset,
            source=args.source,
            h_feats=(
                DIM_BOUND[args.dataset]
                if args.dataset in DIM_BOUND.keys() and args.h_feats > DIM_BOUND[args.dataset]
                else args.h_feats
            ),
            MODEL=args.model,
            BATCH_SIZE=args.batch_size,
            nie=args.nie,
            nrl=args.nrl,
            n_hops=args.n_hops,
            n_layers=args.n_layers,
            lr=args.lr,
            l2_coefs=args.l2_coefs,
            nas_dropout=args.nas_dropout,
            nss_dropout=args.nss_dropout,
            clf_dropout=args.clf_dropout,
        )
    else:
        for source, datasets in DATASETS.items():
            # for source, datasets in [('pyg',['texas']),('cola',['blogcatalog']),('pyg',['cora'])]:
            for dataset in datasets:
                source = source.lower()
                try:
                    main(
                        dataset=dataset,
                        source=source,
                        h_feats=(
                            DIM_BOUND[dataset]
                            if dataset in DIM_BOUND.keys() and args.h_feats > DIM_BOUND[dataset]
                            else args.h_feats
                        ),
                        MODEL=args.model,
                        BATCH_SIZE=args.batch_size,
                        nie=args.nie,
                        nrl=args.nrl,
                        n_hops=args.n_hops,
                    )
                except Exception as e:
                    traceback.print_exc()
                    continue
