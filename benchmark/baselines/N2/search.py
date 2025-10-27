import os
import re
import subprocess
from pathlib import Path

import optuna

# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)

# 参数搜索空间（显式写成变量，便于去重）
search_space = {
    "d_model": [128, 256, 512],
    "nlayers": [1, 2, 4],
    "q_dim": [32, 64],
    "n_q": [4, 8],
    "n_pnode": [128, 256],
    "lr": [0.001, 0.01],
    "lr_step": [0.99999, 1],
    "lr_lb": [0.001],
    "dropout": [0.2, 0.4, 0.6, 0.8],
}

# 保存已使用参数组合
used_combinations = set()


def objective(trial):
    dataset = "photo"  # 可以替换为变量
    source = "pyg"
    gpu = "2"
    trial_id = trial.number  # Optuna 会自动给每个 trial 分配唯一 ID

    # 采样参数
    params = {
        "d_model": trial.suggest_categorical("d_model", search_space["d_model"]),
        "nlayers": trial.suggest_categorical("nlayers", search_space["nlayers"]),
        "q_dim": trial.suggest_categorical("q_dim", search_space["q_dim"]),
        "n_q": trial.suggest_categorical("n_q", search_space["n_q"]),
        "n_pnode": trial.suggest_categorical("n_pnode", search_space["n_pnode"]),
        "lr": trial.suggest_categorical("lr", search_space["lr"]),
        "lr_step": trial.suggest_categorical("lr_step", search_space["lr_step"]),
        "lr_lb": trial.suggest_categorical("lr_lb", search_space["lr_lb"]),
        "dropout": trial.suggest_categorical("dropout", search_space["dropout"]),
    }

    # 去重逻辑（组合转为不可变的 tuple）
    key = tuple(sorted(params.items()))
    if key in used_combinations:
        raise optuna.TrialPruned()  # 跳过重复组合
    used_combinations.add(key)

    # 构建命令
    cmd = [
        "python3",
        "train.py",
        "--dataset",
        dataset,
        "--source",
        source,
        "--trial_number",
        f"{trial_id}",
        "--d_model",
        str(params["d_model"]),
        "--nlayers",
        str(params["nlayers"]),
        "--q_dim",
        str(params["q_dim"]),
        "--n_q",
        str(params["n_q"]),
        "--n_pnode",
        str(params["n_pnode"]),
        "--patience",
        str(100),
        "--lr",
        str(params["lr"]),
        "--lr_step",
        str(params["lr_step"]),
        "--lr_lb",
        str(params["lr_lb"]),
        "--dropout",
        str(params["dropout"]),
        "--cuda_num",
        str(gpu),
    ]

    # 执行命令并捕获输出
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # 保存日志到 logs/{dataset}_{trial_id}.log
    log_path = Path(f"logs/{dataset}/")
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path.joinpath(f"{trial_id}.log"), "w", encoding="utf-8") as f:
        f.write("CMD:\n" + " ".join(cmd) + "\n\n")
        f.write("STDOUT:\n" + result.stdout + "\n\n")
        f.write("STDERR:\n" + result.stderr)

    # 提取输出中的 ACCURACY
    match = re.search(r"optuna_avg_acc_score:\s*([0-9.]+)", result.stdout)
    if match:
        acc = float(match.group(1))
        return -acc  # Optuna 默认最小化，这里负数表示最大化 accuracy
    else:
        # 如果没有正确输出 accuracy，记录失败
        raise ValueError(f"Trial {trial_id} failed to return accuracy.")


# 启动搜索
study = optuna.create_study()
study.optimize(objective, n_trials=128, n_jobs=3)
# squirrel ch 256 15
# blog flickr 128 6

print("Best trial:", study.best_trial)
print(" Value (accuracy):", -study.best_value)
print(" Params:", study.best_params)
