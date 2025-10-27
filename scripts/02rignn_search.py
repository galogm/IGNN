"""
Parameter Search:
```bash
# search on unified 10x random splits
id=1;model="r-IGNN";d=chameleon;gpu=$id;log_path=logs/$model;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=128 --n_jobs=3 > $log_path/$id.log 2>&1 & echo $!

id=0;model="r-IGNN";d=products;gpu=$id;log_path=logs/$model;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=64 --n_jobs=2 --repeat=1 > $log_path/$id.log 2>&1 & echo $!

# search on public splits
id=0;model="r-IGNN";d=chameleon;gpu=$id;log_path=logs/$model/public;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --public True > $log_path/$id.log 2>&1 & echo $!
```
"""

import argparse
import gc
import logging
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import optuna
import torch
from optuna.trial import TrialState


class PaddedLevelFormatter(logging.Formatter):
    """Formatter that supports microseconds in datefmt."""

    MAX_LEVEL_LENGTH = 8

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)

        if datefmt:
            return dt.strftime(datefmt)

        return f"{dt.strftime('%Y-%m-%d %H:%M:%S')},{record.msecs:03d}"

    def format(self, record):
        record.levelname = record.levelname.ljust(self.MAX_LEVEL_LENGTH)
        return super().format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = PaddedLevelFormatter(
    "[%(levelname)s %(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S.%f"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


MODEL = "r-IGNN"

DATASETS = {
    "critical": ["chameleon", "squirrel", "roman-empire", "amazon-ratings"],
    "cola": ["flickr", "blogcatalog"],
    "pyg": ["pubmed", "photo", "actor", "wikics"],
    "linkx": ["arxiv-year", "minesweeper", "pokec"],
    "ogb": ["products", "arxiv"],
}

# fmt: off
search_space = {
    "h_feats": [64, 128, 256, 512],
    "n_layers": [1, 3, 5],
    "n_hops": [1, 3, 6, 8, 10, 16, 32, 64],
    # "agg_type": ["gcn", "gcn_incep", "sage", "gat"],
    "agg_type": ["gcn"],
    # "IN": ["IN-SN", "IN-nSN"],
    "IN": ["IN-nSN"],
    # "RN": ["none", "concat", "attentive", "residual"],
    "RN": ["residual"],
    "preln": [True, False],
    # "fast":  [True, False],

    "norm_type": ["bn", "ln", "none"],
    "act_type": ["relu", "prelu", "none"],
    # "att_act_type": ["tanh", "sigmoid", "relu", "prelu", "leakyrelu", "gelu", "none"],

    "lr": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "l2_coef": [0.0, 1e-5, 5e-5, 1e-4, 5e-4],
    "pre_dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "hid_dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "clf_dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],

    "early_stop": [50, 100, 150, 200],
}
# fmt: on

n_epochs = defaultdict(
    lambda: 3000,
    {
        "products": 300,
        "pokec": 1200,
    },
)

eval_interval = defaultdict(
    lambda: 1,
    {
        "pokec": 5,
        "products": 3,
    },
)

eval_start = defaultdict(
    lambda: 0,
    {
        "pokec": 1200,
        "products": 250,
    },
)


class GPUOutOfMemoryError(torch.cuda.OutOfMemoryError):
    """Custom exception for GPU Out-of-Memory errors."""


class NaNError(ValueError):
    """Custom exception for GPU Out-of-Memory errors."""


class ValueNoneError(ValueError):
    """Return None."""


def find_source_by_dataset(dataset_name):
    for source, datasets_list in DATASETS.items():
        if dataset_name in datasets_list:
            return source
    return None


def wait_for_study_completion(study, poll_interval=60):
    """Wait until all Optuna trials are either COMPLETE or FAIL."""
    cnt = 0
    while True:
        trials = study.trials
        total = len(trials)
        complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
        pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in trials)
        fail = sum(t.state == optuna.trial.TrialState.FAIL for t in trials)
        running = sum(t.state == optuna.trial.TrialState.FAIL for t in trials)
        if total == complete + fail + pruned or running == 0:
            return total, complete, fail
        time.sleep(poll_interval)
        cnt = cnt + 1
        time_waited = cnt * poll_interval / 60
        logger.info("wait %d mins", time_waited)
        if time_waited > 360:
            return None, None, None


def should_retry(flag_file_path: str) -> bool:
    """
    Checks a one-time flag.
    """
    if os.path.exists(flag_file_path):
        return False
    fd = -1
    try:
        fd = os.open(flag_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        logger.info("Retry in process %d", os.getpid())
        return True
    except OSError:
        logger.info("Not Retry in process %d", os.getpid())
        return False
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass


def objective(
    trial: optuna.trial.Trial,
    dataset,
    source,
    gpu,
    log_path,
    prune_fail_pruned=True,
    repeat=None,
    public=False,
):
    trial_id = trial.number

    # fmt: off
    params = {
        "h_feats": trial.suggest_categorical("h_feats", search_space["h_feats"]),
        "n_layers": trial.suggest_categorical("n_layers", search_space["n_layers"]),
        "agg_type": trial.suggest_categorical("agg_type", search_space["agg_type"]),
        "IN": trial.suggest_categorical("IN", search_space["IN"]),
        "RN": trial.suggest_categorical("RN", search_space["RN"]),
        "preln": trial.suggest_categorical("preln", search_space["preln"]),
        "norm_type": trial.suggest_categorical("norm_type", search_space["norm_type"]),
        "act_type": trial.suggest_categorical("act_type", search_space["act_type"]),
        # "att_act_type": trial.suggest_categorical("att_act_type", search_space["att_act_type"]),
        "lr": trial.suggest_categorical("lr", search_space["lr"]),
        "l2_coef": trial.suggest_categorical("l2_coef", search_space["l2_coef"]),
        "pre_dropout": trial.suggest_categorical("pre_dropout", search_space["pre_dropout"]),
        "hid_dropout": trial.suggest_categorical("hid_dropout", search_space["hid_dropout"]),
        "clf_dropout": trial.suggest_categorical("clf_dropout", search_space["clf_dropout"]),
        "early_stop": trial.suggest_categorical("early_stop", search_space["early_stop"]),
    }
    params["n_hops"] = (
        trial.suggest_categorical("n_hops", search_space["n_hops"])
        if params["n_layers"] == 1
        else 1
    )

    # params["fast"] = (
    #     trial.suggest_categorical("fast", search_space["fast"])
    #     if params["n_layers"] == 1
    #     else False
    # )

    cmd = [
        "python",
        "-u",
        "-m",
        "main",
        "--gpu_id", str(gpu),
        "--seed", str(42),
        "--dataset", dataset,
        "--source", source,
        "--model", "ignn",
        "--n_epochs", str(n_epochs[dataset]),
        "--agg_type", params["agg_type"],
        "--IN", params["IN"],
        "--h_feats", str(params["h_feats"]),
        "--lr", str(params["lr"]),
        "--l2_coef", str(params["l2_coef"]),
        "--n_hops", str(params["n_hops"]),
        "--n_layers", str(params["n_layers"]),
        "--early_stop", str(params["early_stop"]),

        "--RN", params["RN"],
        "--norm_type", params["norm_type"],
        "--act_type", params["act_type"],
        # "--att_act_type", params["att_act_type"],

        "--preln", str(params["preln"]),
        # "--fast", str(params["fast"]),

        "--pre_dropout", str(params["pre_dropout"]),
        "--hid_dropout", str(params["hid_dropout"]),
        "--clf_dropout", str(params["clf_dropout"]),

        "--eval_interval", str(eval_interval[dataset]),
        "--eval_start", str(eval_start[dataset]),

        "--public", str(public),
    ]

    if repeat is not None:
        cmd.append("--repeat")
        cmd.append(str(repeat))
    # fmt: on

    logger.info("Trial %d: %s", trial_id, " ".join(cmd))

    states_to_consider = [TrialState.COMPLETE]
    if prune_fail_pruned:
        states_to_consider.extend([TrialState.FAIL, TrialState.PRUNED])
    trials_to_consider = trial.study.get_trials(deepcopy=False, states=tuple(states_to_consider))

    # Prune duplicated trail settings
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            logger.info("Duplicate trial pruned: %s", trial.params)
            raise optuna.exceptions.TrialPruned()

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )

        if "CUDA out of memory" in result.stderr or "CUDA out of memory" in result.stdout:
            logger.error("Trial %d encountered OOM. FAIL.", trial.number)
            trial.set_user_attr("fail_reason", "GPU_OOM")
            original_oom_trial_number = trial.user_attrs.get(
                "original_oom_trial_number", trial.number
            )
            trial.set_user_attr("original_oom_trial_number", original_oom_trial_number)
            raise GPUOutOfMemoryError("CUDA OOM error detected in subprocess.")

        if "contains NaN" in result.stderr or "contains NaN" in result.stdout:
            logger.error("Trial %d encountered NaN. FAIL.", trial.number)
            trial.set_user_attr("fail_reason", "NaN_value")
            original_oom_trial_number = trial.user_attrs.get(
                "original_nan_trial_number", trial.number
            )
            trial.set_user_attr("original_nan_trial_number", original_oom_trial_number)
            raise NaNError("NaN error detected in subprocess.")

        if result.returncode != 0:
            logger.error(
                "Trial %d: Subprocess exited with error code %d.",
                trial.number,
                result.returncode,
            )
            trial.set_user_attr("fail_reason", f"SUBPROCESS_EXIT_CODE_{result.returncode}")
            raise RuntimeError(
                f"Subprocess failed with exit code {result.returncode}. Stderr: {result.stderr.strip()}"
            )

        metric_to_optimize = "Results: 	Acc:"

        clustering_block_match = re.search(
            rf"{re.escape(metric_to_optimize)}(\d+\.\d+)(\s*[+-±]\s*\d+\.\d+)\s*",
            result.stderr + result.stdout,
        )

        metric_value = None
        if clustering_block_match:
            mean_str = clustering_block_match.group(1)
            metric_value = float(mean_str)
        if metric_value is not None:
            logger.info("Trial %d for dataset %s: %f", trial_id, dataset, metric_value)
            return -metric_value / 100.0

        logger.error("Trial %d encountered None Value. FAIL.", trial.number)
        raise ValueNoneError(f"Trial {trial_id} failed with value None.")

    finally:
        torch.cuda.empty_cache()
        gc.collect()
        with open(log_path.joinpath(f"{trial_id}.log"), "w", encoding="utf-8") as f:
            f.write("CMD:\n" + " ".join(cmd) + "\n\n")
            f.write("STDOUT:\n" + result.stdout + "\n\n")
            f.write("STDERR:\n" + result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run your script with specified GPU, metric, dataset, trials, and jobs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--gpu", type=str, default="0", help="The GPU number to use (e.g., 0, 1)")
    parser.add_argument(
        "--metric",
        type=str,
        default="acc",
        choices=["acc"],
        help='The metric type to use ("acc")',
    )
    parser.add_argument(
        "--public",
        type=lambda x: True if x == "True" else False,
        default=False,
        help="If True, use public splits. Otherwise use unified random 10x splits.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="repeat",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "chameleon",
            "squirrel",
            "actor",
            "blogcatalog",
            "flickr",
            "roman-empire",
            "amazon-ratings",
            "pubmed",
            "photo",
            "wikics",
            "arxiv-year",
            "citeseer",
            "cora",
            "minesweeper",
            "arxiv",
            "products",
            "pokec",
        ],
        help="The name of the dataset to process",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=128,
        help="Number of trials for the experiment (default: 256)",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of parallel jobs (default: 1)"
    )

    args = parser.parse_args()

    # Access the parsed arguments
    gpu_number = args.gpu
    metric = args.metric
    d = args.dataset
    n_trials: int = args.n_trials
    n_jobs: int = args.n_jobs
    repeat = args.repeat
    public = args.public

    source_category = find_source_by_dataset(d)

    LOCK_FILE_FLOCK = (
        f"logs/{MODEL}/{d}/trails/{metric}.lock"
        if not public
        else f"logs/{MODEL}/public/{d}/trails/{metric}.lock"
    )
    log_path = (
        Path(f"logs/{MODEL}/{d}/trails") if not public else Path(f"logs/{MODEL}/public/{d}/trails")
    )
    os.makedirs(log_path, exist_ok=True)
    if source_category:
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{log_path}/hyperparams.db",
            engine_kwargs={"connect_args": {"timeout": 100}},
        )
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{MODEL}_train_study_{d}",
            storage=storage,
            load_if_exists=True,
        )

        study.optimize(
            lambda trial: objective(
                trial,
                dataset=d,
                source=source_category,
                gpu=gpu_number,
                log_path=log_path,
                repeat=repeat,
                public=public,
            ),
            n_trials=n_trials,
            n_jobs=n_jobs,
            catch=(
                GPUOutOfMemoryError,
                NaNError,
                optuna.exceptions.TrialPruned,
                ValueNoneError,
            ),
            gc_after_trial=True,
        )

        if should_retry(LOCK_FILE_FLOCK):
            total, complete, fail = wait_for_study_completion(study=study)

            if fail is not None:
                logger.info("Trails: %d", total)
                logger.info("Complete: %d", complete)
                logger.info("Failed: %d", fail)

                trials_to_check = study.get_trials(deepcopy=True)
                trials_enqueued = 0
                for trial in trials_to_check:
                    if (
                        trial.state != optuna.trial.TrialState.FAIL
                        or trial.user_attrs.get("fail_reason") != "GPU_OOM"
                    ):
                        continue
                    study.enqueue_trial(
                        params=trial.params,
                        user_attrs={
                            "is_retried_oom": True,
                            "original_oom_trial_number": trial.number,
                            "is_retry": True,
                        },
                    )
                    trials_enqueued += 1

                if trials_enqueued > 0:
                    logger.info("Enqueued %d trials for retry.", trials_enqueued)
                    study.optimize(
                        lambda trial: objective(
                            trial,
                            dataset=d,
                            source=source_category,
                            gpu=gpu_number,
                            log_path=log_path,
                            prune_fail_pruned=False,
                            repeat=repeat,
                            public=public,
                        ),
                        n_trials=trials_enqueued,
                        n_jobs=1,
                        catch=(
                            GPUOutOfMemoryError,
                            NaNError,
                            optuna.exceptions.TrialPruned,
                            ValueNoneError,
                        ),
                        gc_after_trial=True,
                    )

                logger.info("\n--- Final Study Results ---")
                logger.info("Total number of trials: %d", len(study.trials))
                logger.info(
                    "Total complete trials: %d",
                    len(
                        study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
                    ),
                )
                logger.info(
                    "Total failed trials: %d",
                    len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.FAIL])),
                )

        logger.info("Best trial: %d", study.best_trial._trial_id)
        logger.info("Value (negative %s): %f", metric, study.best_value)
        logger.info("%s: %s", metric, -study.best_value)
        logger.info("Params: %s", study.best_params)

    else:
        logger.critical("Dataset %s not found in any defined source.", d)
