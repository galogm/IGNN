"""Parameter Search:
```bash
id=0;metric=nmi;d=chameleon;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=256 --n_jobs=1 > $path/$metric-$id.log &
```
"""

import argparse
import gc
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import optuna
import torch
from optuna.trial import TrialState

# from the_utils.logger import logger

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


MODEL = "goat"

DATASETS = {
    "critical": ["chameleon", "squirrel", "roman-empire", "amazon-ratings"],
    "cola": ["flickr", "blogcatalog"],
    "pyg": ["pubmed", "photo", "actor", "wikics"],
    "linkx": ["arxiv-year", "minesweeper", "ogbn-arxiv"],
}

# fmt: off
search_space = {
    "lr": [1e-4, 5e-4, 1e-3, 5e-3],  # Explore around the common 1e-3, wider range
    "epochs": [500],  # Default 500, can explore more
    "batch_size": [512, 1024, 2048], # Values seen in examples (512, 1024, 8192)
    "test_batch_size": [256, 1024], # Values seen in examples (256, 8192)
    "sizes": [[20,10,5]], # Fixed as per default, no variation in examples
    "test_sizes": [[20,10,5]], # Fixed as per default, no variation in examples
    "test_freq": [1, 5, 10], # Values seen in examples (1, 5)
    "num_workers": [4, 8], # Consistently 4 in examples, but 8 is also common
    "conv_type": ["full"], # All choices from arg_parser, 'full' seen in examples
    "hidden_dim": [128, 256], # Values seen in examples (128, 256)
    "global_dim": [128], # Default 128, can explore higher
    "num_layers": [1], # Default 1, can explore more
    "num_heads": [2, 4], # Values seen in examples (2, 4)
    "attn_dropout": [0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9], # Default 0, can explore more
    "ff_dropout": [0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9], # Default 0.5, can explore more
    "skip": [0, 1], # Default 0, can explore if skipping layers helps
    "dist_count_norm": [1], # Fixed to default
    "num_centroids": [4096], # Consistently 4096 in examples, but higher might be relevant
    "hetero_train_prop": [0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9], # Explicitly mentioned for arxiv-year and snap-patents
    # The following are flags or fixed values and typically not part of a hyperparameter search:
    # "dataset": [], # Handled by outer loop
    # "source": ["pyg"],
    # "data_root": [],
    # "linkx_data_root": [],
    # "data_downloading_flag": [False],
    # "device": [],
    # "estop_steps": [100],
    # "seed": [42],
    # "runs": [10],
    # "no_bn": [False],
    # "norm_type": ["batch_norm"],
    # "eval": [False],
    # "eval_epoch": [100],
    # "save_ckpt": [False],
}
# fmt: on


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
    trial: optuna.trial.Trial, dataset, source, gpu, metric, log_path, prune_fail_pruned=True
):
    trial_id = trial.number

    # fmt: off
    params = {
        "lr": trial.suggest_categorical("lr", search_space["lr"]),
        "epochs": trial.suggest_categorical("epochs", search_space["epochs"]),
        "batch_size": trial.suggest_categorical("batch_size", search_space["batch_size"]),
        "test_batch_size": trial.suggest_categorical("test_batch_size", search_space["test_batch_size"]),
        "test_freq": trial.suggest_categorical("test_freq", search_space["test_freq"]),
        "num_workers": trial.suggest_categorical("num_workers", search_space["num_workers"]),
        "conv_type": trial.suggest_categorical("conv_type", search_space["conv_type"]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", search_space["hidden_dim"]),
        "global_dim": trial.suggest_categorical("global_dim", search_space["global_dim"]),
        "num_layers": trial.suggest_categorical("num_layers", search_space["num_layers"]),
        "num_heads": trial.suggest_categorical("num_heads", search_space["num_heads"]),
        "attn_dropout": trial.suggest_categorical("attn_dropout", search_space["attn_dropout"]),
        "ff_dropout": trial.suggest_categorical("ff_dropout", search_space["ff_dropout"]),
        "skip": trial.suggest_categorical("skip", search_space["skip"]),
        "num_centroids": trial.suggest_categorical("num_centroids", search_space["num_centroids"]),
        "hetero_train_prop": trial.suggest_categorical("hetero_train_prop", search_space["hetero_train_prop"]),
    }

    cmd = [
        "python",
        "-u",
        "-m",
        "arxiv_ERM_ns",
        "--seed", str(42), # Fixed to default in arg_parser
        "--dataset", dataset, # `dataset` variable needs to be defined in your Optuna study setup
        "--source", source,   # `source` variable needs to be defined in your Optuna study setup
        "--device", str(gpu), # `device` variable needs to be defined in your Optuna study setup (arg_parser uses device, best params uses gpu implied by context)
        "--hetero_train_prop", str(params["hetero_train_prop"]),
        "--runs", str(10),
        "--estop_steps", str(100), # Fixed to default
        "--lr", str(params["lr"]),
        "--epochs", str(params["epochs"]),
        "--batch_size", str(params["batch_size"]),
        "--test_batch_size", str(params["test_batch_size"]),
        "--sizes", *map(str, search_space["sizes"][0]), # Unpack the fixed list
        "--test_sizes", *map(str, search_space["test_sizes"][0]), # Unpack the fixed list
        "--test_freq", str(params["test_freq"]),
        "--num_workers", str(params["num_workers"]),
        "--conv_type", str(params["conv_type"]),
        "--hidden_dim", str(params["hidden_dim"]),
        "--global_dim", str(params["global_dim"]),
        "--num_layers", str(params["num_layers"]),
        "--num_heads", str(params["num_heads"]),
        "--attn_dropout", str(params["attn_dropout"]),
        "--ff_dropout", str(params["ff_dropout"]),
        "--skip", str(params["skip"]),
        "--dist_count_norm", str(1), # Fixed to default
        "--num_centroids", str(params["num_centroids"]),
        # "--no_bn", # Include if you want to set this flag
        "--norm_type", "batch_norm", # Fixed to default
    ]
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

        metric_to_optimize = "Final Test: "

        clustering_block_match = re.search(
            rf"\s*{re.escape(metric_to_optimize)}\s*(\d+\.\d+)(\s*[+-±]\s*\d+\.\d+)\s*",
            result.stdout,
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
        choices=["acc"],
        help='The metric type to use ("acc")',
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
            "ogbn-arxiv",
        ],
        help="The name of the dataset to process",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=256,
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

    source_category = find_source_by_dataset(d)

    LOCK_FILE_FLOCK = f"logs/{d}/{metric}/{metric}.lock"
    log_path = Path(f"logs/{d}/{metric}")
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
                metric=metric,
                log_path=log_path,
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
                            metric=metric,
                            log_path=log_path,
                            prune_fail_pruned=False,
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
