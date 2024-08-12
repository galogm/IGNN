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

torch.set_printoptions(threshold=10_000)
np.set_printoptions(threshold=10_000)

from flatgnn.models import FlatGNN
from flatgnn.modules import Data
from flatgnn.utils import get_splits_mask
from flatgnn.utils import read_configs
from flatgnn.utils import set_args_wrt_dataset

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
    "cora": False,
    "citeseer": False,
    "texas": False,
    "cornell": False,
    "wisconsin": False,
    "minesweeper": False,
    "tolokers": False,
    "questions": False,
    "Johns Hopkins55": False,
    "Reed98": False,
    "Amherst41": False,
    "Cornell5": False,
    "corafull": False,
    "arxiv-year": False,
    "snap-patents": False,
}

feats = {
    "texas": 512,
    "cornell": 512,
    "wisconsin": 512,
    "chameleon": 512,
    "squirrel": 512,
    "actor": 512,
    "cora": 512,
    "citeseer": 512,
    "pubmed": 500,
    "roman-empire": 300,
    "amazon-ratings": 300,
    "minesweeper": 512,
    "tolokers": 512,
    "questions": 512,
    "flickr": 512,
    "blogcatalog": 512,
    "Johns Hopkins55": 512,
    "Reed98": 512,
    "Amherst41": 512,
    "Cornell5": 512,
    "corafull": 512,
    "wikics": 300,
    "photo": 256,
    "arxiv-year": 512,
    "snap-patents": 512,
}

l2_coefs = {
    "texas": 0.00005,
    "cornell": 0.00005,
    "wisconsin": 0.00005,
    "chameleon": 0.00005,
    "squirrel": 0.00005,
    "actor": 0.00005,
    "cora": 0.00005,
    "citeseer": 0.00005,
    "pubmed": 0.00005,
    "roman-empire": 0.00005,
    "amazon-ratings": 0.00005,
    "minesweeper": 0.00005,
    "tolokers": 0.00005,
    "questions": 0.00005,
    "flickr": 0.00005,
    "blogcatalog": 0.00005,
    "Johns Hopkins55": 0.00005,
    "Reed98": 0.00005,
    "Amherst41": 0.00005,
    "Cornell5": 0.00005,
    "corafull": 0.00005,
    "wikics": 0.00005,
    "photo": 0.00005,
    "arxiv-year": 0.00005,
    "snap-patents": 0.00005,
}
bss = {
    "texas": 1024,
    "cornell": 1024,
    "wisconsin": 1024,
    "chameleon": 1024,
    "squirrel": 1024,
    "actor": 1024,
    "cora": 1024,
    "citeseer": 1024,
    "pubmed": 1024,
    "roman-empire": 1024,
    "amazon-ratings": 1024,
    "minesweeper": 1024,
    "tolokers": 1024,
    "questions": 1024,
    "flickr": 1024,
    "blogcatalog": 1024,
    "Johns Hopkins55": 1024,
    "Reed98": 1024,
    "Amherst41": 1024,
    "Cornell5": 1024,
    "corafull": 1024,
    "wikics": 1024,
    "photo": 1024,
    "arxiv-year": 1024,
    "snap-patents": 1024,
}
lrs = {
    "texas": 0.001,
    "cornell": 0.001,
    "wisconsin": 0.001,
    "chameleon": 0.001,
    "squirrel": 0.001,
    "actor": 0.001,
    "cora": 0.001,
    "citeseer": 0.001,
    "pubmed": 0.001,
    "roman-empire": 0.001,
    "amazon-ratings": 0.001,
    "minesweeper": 0.001,
    "tolokers": 0.001,
    "questions": 0.001,
    "flickr": 0.001,
    "blogcatalog": 0.001,
    "Johns Hopkins55": 0.001,
    "Reed98": 0.001,
    "Amherst41": 0.001,
    "Cornell5": 0.001,
    "corafull": 0.001,
    "wikics": 0.001,
    "photo": 0.001,
    "arxiv-year": 0.001,
    "snap-patents": 0.001,
}
ess = {
    "texas": 100,
    "cornell": 100,
    "wisconsin": 100,
    "chameleon": 100,
    "squirrel": 200,
    "actor": 100,
    "cora": 100,
    "citeseer": 200,
    "pubmed": 300,
    "roman-empire": 100,
    "amazon-ratings": 100,
    "flickr": 100,
    "blogcatalog": 100,
    "corafull": 400,
    "wikics": 200,
    "photo": 200,
    "minesweeper": 200,
    "tolokers": 200,
    "questions": 200,
    "Johns Hopkins55": 200,
    "Reed98": 200,
    "Amherst41": 200,
    "Cornell5": 200,
    "arxiv-year": 200,
    "snap-patents": 200,
}
nas_dropouts = {
    "flickr": 0.8,
    "blogcatalog": 0.5,
    "chameleon": 0.8,
    "squirrel": 0.8,
    "actor": 0.0,
    "pubmed": 0.5,
    "photo": 0.4,
    "roman-empire": 0.2,
    "amazon-ratings": 0.0,
    "wikics": 0.2,
    "cora": 0.8,
    "citeseer": 0.8,
    "corafull": 0.4,
    "arxiv-year": 0.5,
    "snap-patents": 0.5,
    "minesweeper": 0.0,
    "tolokers": 0.0,
    "questions": 0.0,
    "Johns Hopkins55": 0.0,
    "Reed98": 0.0,
    "Amherst41": 0.0,
    "Cornell5": 0.0,
    "texas": 0.8,
    "cornell": 0.0,
    "wisconsin": 0.0,
}
nss_dropouts = {
    "flickr": 0.8,
    "blogcatalog": 0.8,
    "chameleon": 0.8,
    "squirrel": 0.8,
    "actor": 0.8,
    "pubmed": 0.5,
    "photo": 0.8,
    "roman-empire": 0.2,
    "amazon-ratings": 0.5,
    "wikics": 0.2,
    "cora": 0.8,
    "citeseer": 0.8,
    "corafull": 0.4,
    "arxiv-year": 0.5,
    "snap-patents": 0.5,
    "minesweeper": 0.0,
    "tolokers": 0.0,
    "questions": 0.0,
    "Johns Hopkins55": 0.0,
    "Reed98": 0.0,
    "Amherst41": 0.0,
    "Cornell5": 0.0,
    "texas": 0.8,
    "cornell": 0.0,
    "wisconsin": 0.0,
}
clf_dropouts = {
    "flickr": 0.9,
    "blogcatalog": 0.9,
    "chameleon": 0.9,
    "squirrel": 0.9,
    "actor": 0.9,
    "photo": 0.9,
    "pubmed": 0.9,
    "roman-empire": 0.2,
    "amazon-ratings": 0.9,
    "wikics": 0.9,
    "cora": 0.9,
    "citeseer": 0.9,
    "corafull": 0.5,
    "arxiv-year": 0.9,
    "snap-patents": 0.9,
    "minesweeper": 0.9,
    "tolokers": 0.9,
    "questions": 0.9,
    "Johns Hopkins55": 0.9,
    "Reed98": 0.9,
    "Amherst41": 0.9,
    "Cornell5": 0.9,
    "texas": 0.9,
    "cornell": 0.9,
    "wisconsin": 0.9,
}
n_hopss = {
    "chameleon": 10,
    "squirrel": 64,
    "actor": 1,
    "flickr": 2,
    "blogcatalog": 10,
    "roman-empire": 1,
    "amazon-ratings": 6,
    "photo": 8,
    "pubmed": 3,
    "wikics": 3,
    "cora": 3,
    "citeseer": 8,
    "texas": 8,
    "cornell": 8,
    "wisconsin": 8,
    "minesweeper": 8,
    "tolokers": 8,
    "questions": 8,
    "Johns Hopkins55": 8,
    "Reed98": 8,
    "Amherst41": 8,
    "Cornell5": 8,
    "corafull": 4,
    "arxiv-year": 8,
    "snap-patents": 8,
}
n_layers = {
    "chameleon": 1,
    "squirrel": 1,
    "actor": 1,
    "flickr": 1,
    "blogcatalog": 1,
    "roman-empire": 5,
    "amazon-ratings": 1,
    "photo": 1,
    "pubmed": 1,
    "wikics": 1,
    "cora": 1,
    "citeseer": 1,
    "texas": 1,
    "cornell": 1,
    "wisconsin": 1,
    "minesweeper": 1,
    "tolokers": 1,
    "questions": 1,
    "Johns Hopkins55": 1,
    "Reed98": 1,
    "Amherst41": 1,
    "Cornell5": 1,
    "corafull": 1,
    "arxiv-year": 1,
    "snap-patents": 1,
}
nrls = {
    "chameleon": "max",
    "squirrel": "concat",
    "actor": "concat",
    "flickr": "lstm",
    "blogcatalog": "concat",
    "roman-empire": "lstm",
    "amazon-ratings": "concat",
    "pubmed": "mean",
    "photo": "concat",
    "wikics": "max",
    "cora": "concat",
    "citeseer": "concat",
    "texas": "concat",
    "cornell": "concat",
    "wisconsin": "concat",
    "minesweeper": "concat",
    "tolokers": "concat",
    "questions": "concat",
    "Johns Hopkins55": "concat",
    "Reed98": "concat",
    "Amherst41": "concat",
    "Cornell5": "concat",
    "corafull": "concat",
    "arxiv-year": "concat",
    "snap-patents": "concat",
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
    "cora": "relu",
    "citeseer": "relu",
    "texas": "relu",
    "cornell": "relu",
    "wisconsin": "relu",
    "minesweeper": "relu",
    "tolokers": "relu",
    "questions": "relu",
    "Johns Hopkins55": "relu",
    "Reed98": "relu",
    "Amherst41": "relu",
    "Cornell5": "relu",
    "corafull": "relu",
    "arxiv-year": "relu",
    "snap-patents": "relu",
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
    "cora": True,
    "citeseer": True,
    "texas": True,
    "cornell": True,
    "wisconsin": True,
    "minesweeper": True,
    "tolokers": True,
    "questions": True,
    "Johns Hopkins55": True,
    "Reed98": True,
    "Amherst41": True,
    "Cornell5": True,
    "corafull": True,
    "arxiv-year": True,
    "snap-patents": True,
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
    "cora": 3,
    "citeseer": 3,
    "texas": 3,
    "cornell": 3,
    "wisconsin": 3,
    "minesweeper": 3,
    "tolokers": 3,
    "questions": 3,
    "Johns Hopkins55": 3,
    "Reed98": 3,
    "Amherst41": 3,
    "Cornell5": 3,
    "corafull": 3,
    "arxiv-year": 3,
    "snap-patents": 3,
}

TRAIN_RATIO = 48
VALID_RATIOS = 32


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


# DATASETS = {
#     "critical": [
#         # 890
#         "chameleon",
#         # 2,223
#         "squirrel",
#     ],
#     "pyg": [
#         "actor",
#         "pubmed",
#     ],
# }

DATASETS = {
    "critical":
        [
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
    "pyg":
        [
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
    # "linkx": [
    #     # (array([0, 1, 2]), array([ 97, 504, 361]))
    #     # 962
    #     # "Reed98",
    #     # # array([0, 1, 2]), array([ 418, 2153, 2609]
    #     # # 5,180
    #     # "Johns Hopkins55",
    #     # # (array([0, 1, 2]), array([ 203, 1015, 1017]))
    #     # # 2,235
    #     # "Amherst41",
    #     # # (array([0, 1, 2]), array([1838, 8135, 8687]))
    #     # # 18,660
    #     # "Cornell5",
    #     # # 41,554
    #     # "Penn94",
    #     # # 168,114
    #     # "twitch-gamers",
    #     # 169,343
    #     "arxiv-year",
    #     # 421,961
    #     # "genius",
    #     # # 1,632,803
    #     # "pokec",
    #     # 2,923,922
    #     "snap-patents",
    # ],
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
):
    BATCH_SIZE = BATCH_SIZE or bss[dataset]
    graph, label, n_clusters = load_data(
        dataset_name=dataset,
        directory=DATA.DATA_DIR,
        source=source,
        row_normalize=norms[dataset],
        rm_self_loop=False,
        add_self_loop=True,
        to_simple=True,
        verbosity=3,
    )

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

    N_HOPS = n_hops if n_hops is not None else n_hopss[dataset]
    params = {
        "nie": nie,
        "nrl": nrl if nrl is not None else nrls[dataset],
        "lr": lrs[dataset],
        "h_feats": h_feats,
        "l2_coef": l2_coefs[dataset],
        "nas_dropout": nas_dropouts[dataset],
        "nss_dropout": nss_dropouts[dataset],
        "clf_dropout": clf_dropouts[dataset],
        "n_epochs": 2000,
        "n_hops": N_HOPS,
        "n_layers": n_layers[dataset],
        "early_stop": ess[dataset],
        "n_intervals": min(n_intervalss[dataset], N_HOPS + 1),
        "act": acts[dataset],
        "layer_norm": layer_norms[dataset],
    }
    params_all = {
        "row_normalized": norms[dataset],
        **params,
    }
    tab_printer(params_all)

    t_start = time.time()

    repeat = 10

    res_list_acc_joint = []
    for i in range(repeat):
        graph.split = i
        # model.load_state_dict(torch.load(STATE, map_location=DEVICE))
        model = FlatGNN(
            in_feats=graph.ndata["feat"].shape[1],
            n_clusters=n_clusters,
            device=DEVICE,
            **params,
        )

        train_mask, val_mask, test_mask = get_splits_mask(
            graph=graph,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIOS,
            repeat=10,
            split_id=i,
            SPLIT_DIR=DATA.SPLIT_DIR,
        )

        model.fit(
            graph=graph,
            labels=label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            split_id=i,
            bs=BATCH_SIZE,
            device=DEVICE,
        )

        with torch.no_grad():
            model.train(False)
            embeddings = model(
                graph.ndata["feat"],
                graph=graph,
                device=DEVICE,
            )
            logits_onehot = model.classifier(embeddings)
            y_pred = torch.argmax(logits_onehot, dim=1)

        acc = ACC(label[test_mask].cpu(), y_pred[test_mask].cpu())
        res_list_acc_joint.append(acc)
        print(f"{graph.name} {i} Acc: {acc}\n\n")

    elapsed_time = f"{(time.time() - t_start)/repeat:.2f}"
    acc_jl = f"{np.array(res_list_acc_joint).mean() * 100:.2f}±{np.array(res_list_acc_joint).std() * 100:.2f}"

    print(f"\nResults: \tAcc:{acc_jl} \tTrain cost: {elapsed_time}s")
    save_to_csv_files(
        results={
            "acc_hl": acc_jl,
        },
        insert_info={
            "dataset": dataset,
        },
        append_info={
            "args": params_all,
            "time": elapsed_time,
            "bs": BATCH_SIZE,
            "source": source,
            "model": MODEL,
        },
        csv_name=f"results_v{VERSION}.csv",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="FlatGNN",
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
                DIM_BOUND[args.dataset] if args.dataset in DIM_BOUND.keys() and
                args.h_feats > DIM_BOUND[args.dataset] else args.h_feats
            ),
            MODEL=args.model,
            BATCH_SIZE=args.batch_size,
            nie=args.nie,
            nrl=args.nrl,
            n_hops=args.n_hops,
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
                            DIM_BOUND[dataset] if dataset in DIM_BOUND.keys() and
                            args.h_feats > DIM_BOUND[dataset] else args.h_feats
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
