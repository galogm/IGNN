"""IGNN"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals,invalid-name,too-many-statements
import random
import time

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
    repeats,
    self_loop_attentive,
)
from ignn.models import IGNN
from ignn.modules import DataConf, INConf
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
    return_type="dgl",
):
    if return_type == "dgl":
        graph, label, n_clusters = load_data(
            dataset_name=dataset,
            directory=DATA_INFO.DATA_DIR,
            source=source,
            row_normalize=norms[dataset],
            rm_self_loop=False if RN != "attentive" else (not self_loop_attentive[dataset]),
            add_self_loop=True if RN != "attentive" else self_loop_attentive[dataset],
            # products and arxiv-year are already simple graphs
            to_simple=dataset not in ["products", "arxiv-year"],
            verbosity=1,
            return_type=return_type,
        )
        n_nodes = graph.num_nodes()
        edge_index = graph.edges()
        features = graph.ndata["feat"]
        name = graph.name
    else:
        data = load_data(
            dataset_name=dataset,
            directory=DATA_INFO.DATA_DIR,
            source=source,
            row_normalize=norms[dataset],
            rm_self_loop=False if RN != "attentive" else (not self_loop_attentive[dataset]),
            add_self_loop=True if RN != "attentive" else self_loop_attentive[dataset],
            # products and arxiv-year are already simple graphs
            to_simple=dataset not in ["products", "arxiv-year"],
            verbosity=1,
            return_type=return_type,
        )
        label = data.y
        n_clusters = data.num_classes
        n_nodes = data.num_nodes
        edge_index = data.edge_index
        features = data.x
        name = data.name

    labeled_idx = torch.where(label != -1)[0] if dataset in ["wiki", "Penn94", "pokec"] else None
    n_clusters = 2 if dataset == "pokec" else n_clusters
    n_clusters = n_clusters if dataset not in ["proteins"] else label.shape[1]

    repeat = repeats[dataset.lower()]

    N_HOPS = n_hops if n_hops is not None else n_hopss[dataset]
    N_LAYERS = (
        n_layers
        if n_layers is not None
        else n_layerss[dataset] if RN not in ["residual", "attentive"] else 1
    )
    LR = lr if lr is not None else lrs[dataset]
    COEF = l2_coef if l2_coef is not None else l2_coefs[dataset]
    DNAS = nas_dropout if nas_dropout is not None else nas_dropouts[dataset]
    DNSS = nss_dropout if nss_dropout is not None else nss_dropouts[dataset]
    DCLF = clf_dropout if clf_dropout is not None else clf_dropouts[dataset]
    IN_config = INConf(
        n_hops=N_HOPS,
        add_self_loop=True,
        remove_self_loop=False,
        symm_norm=True,
        row_normalized=norms[dataset],
        fast=False,
        name=name,
    )
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
        "bs": BATCH_SIZE,
        "eval_start": eval_start,
        "IN_config": IN_config,
        **params,
    }
    tab_printer({**params_all})

    t_start = time.time()

    seed_list = [random.randint(0, 99999) for i in range(repeat)]

    res_list_acc_joint = []

    tms = {"model": "IGNN", "dataset": dataset, "hops": N_HOPS}
    ts = []

    for i in range(repeat):
        # set_seed(seed_list[i])
        model = IGNN(in_feats=features.shape[1], n_clusters=n_clusters, device=DEVICE, **params)

        train_mask, val_mask, test_mask = get_splits(
            graph.ndata if return_type == "dgl" else data,
            name,
            n_nodes,
            i,
            TRAIN_RATIO=TRAIN_RATIO,
            VALID_RATIO=VALID_RATIO,
            DATA=DATA_INFO,
            labeled_idx=labeled_idx,
        )

        tm = model.fit(
            edge_index=edge_index,
            features=features,
            labels=label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            bs=BATCH_SIZE,
            IN_config=IN_config,
            device=DEVICE,
            eval_start=eval_start,
        )

        with torch.no_grad():
            model.train(False)
            if BATCH_SIZE is not None:
                pred_stack = []
                idx = torch.LongTensor(list(range(n_nodes)))
                for _, step in enumerate(range(0, n_nodes, BATCH_SIZE)):
                    batch_idx = idx[step : step + BATCH_SIZE]

                    embeddings = model(
                        edge_index=edge_index,
                        features=features,
                        IN_config=IN_config,
                        batch_idx=batch_idx,
                        device=DEVICE,
                    )
                    logits = model.classifier(embeddings)
                    pred_stack.append(logits)

                y_pred = torch.cat(pred_stack, dim=0)
                train_acc, val_acc, res = metric(
                    name, y_pred, label, train_mask, val_mask, test_mask
                )
            else:
                embeddings = model(
                    edge_index=edge_index,
                    features=features,
                    IN_config=IN_config,
                    device=DEVICE,
                )
                y_pred = model.classifier(embeddings)

                train_acc, val_acc, res = metric(
                    name, y_pred, label, train_mask, val_mask, test_mask
                )

        tms[f"{i}"] = tm
        ts.append(tm)

        res_list_acc_joint.append(res)
        print(f"{name} {i} res: {res}\n\n")

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
        results={"acc_hl": acc_jl, "hop": N_HOPS},
        insert_info={"dataset": dataset},
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

    args = parse_ignn_args()

    TRAIN_RATIO = 48
    VALID_RATIO = 32
    VERSION = args.version
    DEVICE = set_device(str(args.gpu_id))
    set_seed(args.seed)
    DATA_INFO = DataConf(**read_configs("data"))

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
        return_type=args.return_type,
    )
