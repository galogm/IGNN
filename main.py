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
from tqdm import tqdm

from configs.params import (
    RNs,
    acts,
    clf_dropouts,
    ess,
    feats,
    l2_coefs,
    lrs,
    n_hopss,
    n_layerss,
    nas_dropouts,
    norms,
    norms_att,
    nss_dropouts,
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
    nas_dropout=0.0,
    nss_dropout=0.0,
    clf_dropout=0.0,
    early_stop=None,
    num_parts=3,
    eval_interval=1,
    eval_start=0,
    norm=None,
    act_att="tanh",
    TRAIN_RATIO=48,
    VALID_RATIO=32,
    VERSION="1.0",
    device=torch.device("cpu"),
):
    DATA_INFO = DataConf(**read_configs("data"))
    data = load_data(
        dataset_name=dataset,
        source=source,
        directory=DATA_INFO.DATA_DIR,
        row_normalize=pre_norms[dataset],
        rm_self_loop=False if RN != "attentive" else (not self_loop_attentive[dataset]),
        add_self_loop=True if RN != "attentive" else self_loop_attentive[dataset],
        to_simple=dataset not in ["products", "arxiv-year"],
        verbosity=3,
        return_type="pyg",
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
    BATCH_LOAD = ["products_ogb", "pokec_linkx"]
    FAST = name not in BATCH_LOAD
    IN_config = INConf(
        n_hops=N_HOPS,
        add_self_loop=True if RN != "attentive" else self_loop_attentive[dataset],
        remove_self_loop=False if RN != "attentive" else (not self_loop_attentive[dataset]),
        symm_norm=True,
        row_normalized=pre_norms[dataset],
        fast=FAST,
        name=name,
    )

    params = {
        "IN": IN,
        "RN": RN if RN is not None else RNs[dataset],
        "lr": LR,
        "h_feats": h_feats if h_feats is not None else feats[dataset],
        "l2_coef": COEF,
        "nas_dropout": DNAS,
        "nss_dropout": DNSS,
        "clf_dropout": DCLF,
        "n_epochs": n_epochs,
        "n_hops": N_HOPS,
        "n_layers": N_LAYERS,
        "early_stop": early_stop or ess[dataset],
        "act": acts[dataset],
        "norm": (
            None
            if norm is False
            else norm or (norms_att[dataset] if RN == "attentive" else norms[dataset])
        ),
        "loss": "ce" if dataset not in ["proteins"] else "bce",
        "act_att": act_att,
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
    for i in range(repeat):
        model = IGNN(in_feats=features.shape[1], n_clusters=n_clusters, device=device, **params)

        train_mask = val_mask = test_mask = train_loader = test_loader = None
        if name in BATCH_LOAD:
            if name == "pokec_linkx":
                data["train_mask"], data["val_mask"], data["test_mask"] = get_splits(
                    data,
                    name,
                    n_nodes,
                    i,
                    TRAIN_RATIO=TRAIN_RATIO,
                    VALID_RATIO=VALID_RATIO,
                    DATA=DATA_INFO,
                    labeled_idx=labeled_idx,
                )
            train_loader = RandomNodeLoader(data, num_parts=num_parts, shuffle=True, num_workers=5)
            test_loader = RandomNodeLoader(data, num_parts=1, num_workers=5)
        else:
            train_mask, val_mask, test_mask = get_splits(
                data,
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
            IN_config=IN_config,
            train_loader=train_loader,
            test_loader=test_loader,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            device=device,
            eval_interval=eval_interval,
            eval_start=eval_start,
        )

        with torch.no_grad():
            model.train(False)
            if test_loader is not None:
                device_t = torch.device("cpu") if len(test_loader) == 1 else device
                transform = T.Compose([T.ToDevice(device_t), T.ToSparseTensor()])
                model = model.to(device_t)
                y_true = {"train": [], "val": [], "test": []}
                y_pred = {"train": [], "val": [], "test": []}
                for data_t in tqdm(test_loader, "test step"):
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

    save_to_csv_files(
        results={**tms},
        append_info={
            "mean": f"{np.array(ts).mean():.2f}±{np.array(ts).std():.2f}",
        },
        csv_name="times.csv",
    )
    elapsed_time = f"{(time.time() - t_start)/repeat:.2f}"
    acc_jl = f"{np.array(res_list_acc_joint).mean() * 100:.2f}±{np.array(res_list_acc_joint).std() * 100:.2f}"

    logger.info("%s", f"Results: \tAcc:{acc_jl} \tTrain cost: {elapsed_time}s")
    save_to_csv_files(
        results={"acc_hl": acc_jl, "hop": N_HOPS},
        insert_info={"dataset": dataset},
        append_info={
            "args": params_all,
            "time": elapsed_time,
            "source": source,
            "model": MODEL,
        },
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
        nas_dropout=args.nas_dropout,
        nss_dropout=args.nss_dropout,
        clf_dropout=args.clf_dropout,
        early_stop=args.early_stop,
        num_parts=args.num_parts,
        eval_interval=args.eval_interval,
        eval_start=args.eval_start,
        act_att=args.act_att,
        norm=args.norm,
        TRAIN_RATIO=48,
        VALID_RATIO=32,
        VERSION=args.version,
        device=DEVICE,
    )
