"""IGNN"""

# pylint: disable=unused-import,line-too-long,unused-argument,too-many-locals,invalid-name,too-many-statements
import logging
import time

import numpy as np
import torch
import torch_geometric.transforms as T
from graph_datasets import load_data
from the_utils import save_to_csv_files, set_device, set_seed, tab_printer
from torch_geometric.loader import RandomNodeLoader

from configs.params import (
    RNs,
    acts,
    att_norms,
    clf_dropouts,
    ess,
    feats,
    hid_dropouts,
    l2_coefs,
    lrs,
    n_hopss,
    n_layerss,
    norms,
    pre_dropouts,
    pre_norms,
    repeats,
    self_loop_attentive,
)
from ignn.configs import DataConf, INConf
from ignn.models import IGNN
from ignn.utils import metric
from utils import get_splits, parse_ignn_args, read_configs

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)1.1s %(asctime)s %(name)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main(
    dataset,
    source,
    h_feats,
    MODEL,
    IN,
    RN,
    n_hops,
    n_layers=1,
    n_epochs=2000,
    lr=0.001,
    l2_coef=5e-5,
    pre_dropout=None,
    hid_dropout=None,
    clf_dropout=None,
    early_stop=None,
    num_parts=3,
    eval_interval=1,
    eval_start=0,
    norm_type=None,
    att_act_type="tanh",
    act_type="relu",
    agg_type="gcn_incep",
    fast=None,
    pre_lin=None,
    TRAIN_RATIO=48,
    VALID_RATIO=32,
    repeat=None,
    public=False,
    VERSION="1.0",
    device=torch.device("cpu"),
):
    DATA_INFO = DataConf(**read_configs("data"))
    add_self_loop = RN != "attentive" or self_loop_attentive[dataset]
    rm_self_loop = False if RN != "attentive" else (not self_loop_attentive[dataset])
    simple = dataset not in ["products", "arxiv-year"]
    row_norm = pre_norms[dataset]

    data = load_data(
        dataset, DATA_INFO.DATA_DIR, 3, source, "pyg", row_norm, rm_self_loop, add_self_loop, simple
    )
    label = data.y
    n_clusters = data.num_classes
    n_nodes = data.num_nodes
    edge_index = data.edge_index
    features = data.x
    name = data.name
    labeled_idx = torch.where(label != -1)[0] if dataset in ["wiki", "Penn94", "pokec"] else None
    n_clusters = (
        label.shape[1] if dataset == "proteins" else (2 if dataset == "pokec" else n_clusters)
    )

    BATCH_LOAD = ["products_ogb", "pokec_linkx"]
    N_HOPS = n_hops or n_hopss[dataset]
    N_LAYERS = n_layers or n_layerss[dataset]
    FAST = N_LAYERS == 1 and (fast if fast is not None else (name not in BATCH_LOAD))
    IN_config = INConf(name, N_HOPS, add_self_loop, rm_self_loop, True, row_norm, FAST)

    LR = lr if lr is not None else lrs[dataset]
    COEF = l2_coef if l2_coef is not None else l2_coefs[dataset]
    DNAS = pre_dropout if pre_dropout is not None else pre_dropouts[dataset]
    DNSS = hid_dropout if hid_dropout is not None else hid_dropouts[dataset]
    DCLF = clf_dropout if clf_dropout is not None else clf_dropouts[dataset]
    PRE_LN = pre_lin if pre_lin is not None else (name in BATCH_LOAD and RN == "concat")
    params = {
        "h_feats": h_feats or feats[dataset],
        "n_epochs": n_epochs,
        "lr": LR,
        "l2_coef": COEF,
        "early_stop": early_stop or ess[dataset],  # early_stop or
        "pre_dropout": DNAS,
        "hid_dropout": DNSS,
        "clf_dropout": DCLF,
        "n_hops": N_HOPS,
        "IN": IN,
        "RN": RN or RNs[dataset],
        "n_layers": N_LAYERS,
        "loss": "ce" if dataset not in ["proteins"] else "bce",
        "fast": IN_config.fast,
        "pre_lin": PRE_LN,
        "norm_type": norm_type or (att_norms[dataset] if RN == "attentive" else norms[dataset]),
        "agg_type": agg_type,
        "act_type": act_type or acts[dataset],  # act_type or
        "att_act_type": att_act_type,
    }
    params_all = {
        "eval_start": eval_start,
        "eval_interval": eval_interval,
        "IN_config": IN_config,
        **params,
    }
    logger.info("\n%s", tab_printer({**params_all}, verbose=0))

    t_start = time.time()
    res_list_acc_joint = []
    tms = {"model": f"IGNN-{IN}-{RN}", "dataset": dataset, "hops": N_HOPS}
    ts = []
    repeat = repeat if repeat is not None else repeats[dataset.lower()]
    for i in range(repeat):
        model = IGNN(in_feats=features.shape[1], n_clusters=n_clusters, device=device, **params)

        train_mask = val_mask = test_mask = train_loader = test_loader = None
        if name in BATCH_LOAD:
            if name == "pokec_linkx":
                data["train_mask"], data["val_mask"], data["test_mask"] = get_splits(
                    data, name, n_nodes, i, TRAIN_RATIO, VALID_RATIO, DATA_INFO, labeled_idx, public
                )
            train_loader = RandomNodeLoader(data, num_parts=num_parts, shuffle=True, num_workers=5)
            test_loader = RandomNodeLoader(data, num_parts=1, num_workers=5)
        else:
            train_mask, val_mask, test_mask = get_splits(
                data,
                name,
                n_nodes,
                i,
                repeat,
                TRAIN_RATIO,
                VALID_RATIO,
                DATA_INFO,
                labeled_idx,
                public,
            )

        tm = model.fit(
            edge_index,
            features,
            label,
            IN_config,
            train_loader,
            test_loader,
            train_mask,
            val_mask,
            test_mask,
            eval_interval,
            eval_start,
            save_state=False,
            device=device,
        )

        with torch.no_grad():
            model.train(False)
            if test_loader is not None:
                device_t = torch.device("cpu") if len(test_loader) == 1 else device
                transform = T.Compose([T.ToDevice(device_t), T.ToSparseTensor()])
                model = model.to(device_t)
                y_true = {"train": [], "val": [], "test": []}
                y_pred = {"train": [], "val": [], "test": []}
                for _, data_t in enumerate(test_loader):
                    data_t = transform(data_t)
                    embeddings = model(data_t.adj_t, data_t.x, IN_config, device_t)
                    logits = model.classifier(embeddings)
                    for split in ["train", "val", "test"]:
                        mask = data_t[f"{split}_mask"]
                        y_true[split].append(data_t.y[mask].cpu())
                        y_pred[split].append(logits[mask].cpu())
                test_acc = metric(name, torch.cat(y_pred["test"]), torch.cat(y_true["test"]))
            else:
                embeddings = model(edge_index.to(device), features.to(device), IN_config, device)
                y_pred = model.classifier(embeddings)
                _, _, test_acc = metric(name, y_pred, label, train_mask, val_mask, test_mask)

        tms[f"{i}"] = tm
        ts.append(tm)
        res_list_acc_joint.append(test_acc)
        logger.info("%s", f"{name} {i} res: {test_acc}\n\n")

    elapsed_time = f"{(time.time() - t_start)/repeat:.2f}"
    acc_jl = f"{np.array(res_list_acc_joint).mean() * 100:.2f}±{np.array(res_list_acc_joint).std() * 100:.2f}"
    logger.info("%s", f"Results: \tAcc:{acc_jl} \tTrain cost: {elapsed_time}s")
    save_to_csv_files(
        results={**tms},
        append_info={"mean": f"{np.array(ts).mean():.2f}±{np.array(ts).std():.2f}"},
        csv_name="times.csv",
    )
    save_to_csv_files(
        results={"acc_hl": acc_jl, "hop": N_HOPS},
        insert_info={"dataset": dataset},
        append_info={"args": params_all, "time": elapsed_time, "source": source, "model": MODEL},
        csv_name=f"results_v{VERSION}.csv",
    )


if __name__ == "__main__":

    args = parse_ignn_args()
    DEVICE = set_device(str(args.gpu_id))
    set_seed(args.seed)

    main(
        dataset=args.dataset,
        source=args.source,
        h_feats=args.h_feats,
        MODEL=args.model,
        IN=args.IN,
        RN=args.RN,
        n_hops=args.n_hops,
        n_layers=args.n_layers,
        n_epochs=args.n_epochs,
        lr=args.lr,
        l2_coef=args.l2_coef,
        pre_dropout=args.pre_dropout,
        hid_dropout=args.hid_dropout,
        clf_dropout=args.clf_dropout,
        early_stop=args.early_stop,
        num_parts=args.num_parts,
        eval_interval=args.eval_interval,
        eval_start=args.eval_start,
        att_act_type=args.att_act_type,
        act_type=args.act_type,
        norm_type=args.norm_type,
        agg_type=args.agg_type,
        fast=args.fast,
        pre_lin=args.preln,
        TRAIN_RATIO=48,
        VALID_RATIO=32,
        repeat=args.repeat,
        public=args.public,
        VERSION=args.version,
        device=DEVICE,
    )
