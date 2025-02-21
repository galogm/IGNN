"""Full Batch"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals
import random
import time
import traceback

import numpy as np
import torch
from graph_datasets import load_data
from the_utils import save_to_csv_files, set_device, set_seed, tab_printer

from ignn.configs.params import (
    RNs,
    acts,
    clf_dropouts,
    ess,
    feats,
    l2_coefs,
    layer_norms,
    layer_norms_att,
    lrs,
    n_hopss,
    n_layerss,
    nas_dropouts,
    norms,
    nss_dropouts,
    self_loop_attentive,
)
from ignn.models import IGNN
from ignn.modules import Data
from ignn.utils import get_splits, metric, parse_ignn_args, read_configs

torch.set_printoptions(threshold=10_000)
np.set_printoptions(threshold=10_000)


def main(
    dataset,
    source,
    h_feats,
    MODEL,
    BATCH_SIZE=None,
    IN=None,
    RN=None,
    n_hops=None,
    n_layers=None,
    lr=None,
    l2_coef=None,
    nas_dropout=None,
    nss_dropout=None,
    clf_dropout=None,
    eval_start=1,
):
    graph, label, n_clusters = load_data(
        dataset_name=dataset,
        directory=DATA.DATA_DIR,
        source=source,
        row_normalize=norms[dataset],
        rm_self_loop=False if RN != "attentive" else (not self_loop_attentive[dataset]),
        add_self_loop=True if RN != "attentive" else self_loop_attentive[dataset],
        # products and arxiv-year are already simple graphs
        to_simple=True if dataset not in ["products", "arxiv-year"] else False,
        verbosity=3,
    )

    repeat = (
        {
            "pokec": 5,
            "arxiv": 3,
            "products": 3,
        }[dataset] if dataset in [
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
        ] else 10
    )

    labeled_idx = None
    if dataset in ["wiki", "Penn94", "pokec"]:
        labeled_idx = torch.where(label != -1)[0]
        n_clusters = 2 if dataset == "pokec" else n_clusters

    N_HOPS = n_hops if n_hops is not None else n_hopss[dataset]
    N_LAYERS = (
        n_layers if n_layers is not None else
        n_layerss[dataset] if RN not in ["residual", "attentive"] else 1
    )
    LR = lr if lr is not None else lrs[dataset]
    COEF = l2_coef if l2_coef is not None else l2_coefs[dataset]
    DNAS = nas_dropout if nas_dropout is not None else nas_dropouts[dataset]
    DNSS = nss_dropout if nss_dropout is not None else nss_dropouts[dataset]
    DCLF = clf_dropout if clf_dropout is not None else clf_dropouts[dataset]
    params = {
        "IN": IN,
        "RN": RN if RN is not None else RNs[dataset],
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
        "act": acts[dataset],
        "layer_norm": layer_norms[dataset] if RN != "attentive" else layer_norms_att[dataset],
        "loss": "ce" if dataset not in ["proteins"] else "bce",
        "transform_first": False,
    }
    params_all = {
        "row_normalized": norms[dataset],
        "bs": BATCH_SIZE,
        "eval_start": eval_start,
        **params,
    }
    tab_printer({**params_all})

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
        model = IGNN(
            in_feats=graph.ndata["feat"].shape[1],
            n_clusters=n_clusters if dataset not in ["proteins"] else label.shape[1],
            device=DEVICE,
            **params,
        )

        train_mask, val_mask, test_mask = get_splits(
            graph,
            label,
            i,
            TRAIN_RATIO=TRAIN_RATIO,
            VALID_RATIO=VALID_RATIO,
            DATA=DATA,
            labeled_idx=labeled_idx,
        )

        tm = model.fit(
            graph=graph,
            labels=label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            split_id=i,
            bs=BATCH_SIZE,
            device=DEVICE,
            eval_start=eval_start,
        )

        with torch.no_grad():
            if BATCH_SIZE is not None:
                model.train(False)
                pred_stack = []
                idx = torch.LongTensor(list(range(graph.num_nodes())))
                for _, step in enumerate(range(0, graph.num_nodes(), BATCH_SIZE)):
                    batch_idx = idx[step:step + BATCH_SIZE]

                    embeddings = model(
                        batch_idx=batch_idx,
                        graph=graph,
                        device=DEVICE,
                    )
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
            else:
                model.train(False)
                embeddings = model(
                    graph=graph,
                    device=DEVICE,
                )
                y_pred = model.classifier(embeddings)

                train_acc, val_acc, res = metric(
                    graph.name,
                    logits=y_pred,
                    labels=label,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                )

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
                # "coRNell",
                # "wisconsin",
                # "corafull",
                # "cora",
                # "citeseer",
                "photo",
                "actor",
                "pubmed",
                "wikics",
            ],
        "Critical":
            [
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
        "linkx":
            [
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
                "CoRNell5",
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
    TRAIN_RATIO = 48
    VALID_RATIO = 32

    args = parse_ignn_args()

    VERSION = args.version
    DEVICE = set_device(str(args.gpu_id))
    set_seed(args.seed)

    DATA = Data(**read_configs("data"))

    if args.dataset != "all":
        main(
            dataset=args.dataset,
            source=args.source,
            h_feats=args.h_feats,
            MODEL=args.model,
            BATCH_SIZE=args.batch_size,
            IN=args.IN,
            RN=args.RN,
            n_hops=args.n_hops,
            n_layers=args.n_layers,
            lr=args.lr,
            l2_coef=args.l2_coef,
            nas_dropout=args.nas_dropout,
            nss_dropout=args.nss_dropout,
            clf_dropout=args.clf_dropout,
            eval_start=args.eval_start,
        )
    else:
        for source, datasets in DATASETS.items():
            for dataset in datasets:
                source = source.lower()
                try:
                    main(
                        dataset=dataset,
                        source=source,
                        h_feats=args.h_feats,
                        MODEL=args.model,
                        BATCH_SIZE=args.batch_size,
                        IN=args.IN,
                        RN=args.RN,
                        n_hops=args.n_hops,
                        n_layers=args.n_layers,
                        lr=args.lr,
                        l2_coef=args.l2_coef,
                        nas_dropout=args.nas_dropout,
                        nss_dropout=args.nss_dropout,
                        clf_dropout=args.clf_dropout,
                        eval_start=args.eval_start,
                    )
                except Exception as e:
                    traceback.print_exc()
                    continue
