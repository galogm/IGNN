import os
import re
import subprocess
from pathlib import Path

import optuna

# 确保 logs 目录存在
os.makedirs("logs", exist_ok=True)

# 参数搜索空间（显式写成变量，便于去重）
search_space = {
    "hid": [128, 256, 512],
    "nlayers": [3, 4],
    "lr1": [0.1, 0.15, 0.01, 0.05],
    "lr2": [0.5, 0.005],
    "wd1": [0, 5e-4, 5e-5],
    "wd2": [0, 5e-4, 1e-4],
    "dpC": [0, 0.3, 0.6, 0.9],
    "dpM": [0, 0.4, 0.6, 0.8],
    "tau": [1, 0.3, 0.6, 0.9],
    "dropout": [0.2, 0.4, 0.6, 0.8],
    "bias": ["none", "bn"],
}

# 保存已使用参数组合
used_combinations = set()


def objective(trial):
    dataset = "flickr"  # 可以替换为变量
    source = "cola"
    gpu = "3"
    trial_id = trial.number  # Optuna 会自动给每个 trial 分配唯一 ID


    # 采样参数
    params = {
        "hid": trial.suggest_categorical("hid", search_space["hid"]),
        "nlayers": trial.suggest_categorical("nlayers", search_space["nlayers"]),
        "lr1": trial.suggest_categorical("lr1", search_space["lr1"]),
        "lr2": trial.suggest_categorical("lr2", search_space["lr2"]),
        "wd1": trial.suggest_categorical("wd1", search_space["wd1"]),
        "wd2": trial.suggest_categorical("wd2", search_space["wd2"]),
        "dpC": trial.suggest_categorical("dpC", search_space["dpC"]),
        "dpM": trial.suggest_categorical("dpM", search_space["dpM"]),
        "tau": trial.suggest_categorical("tau", search_space["tau"]),
        "dropout": trial.suggest_categorical("dropout", search_space["dropout"]),
        "bias": trial.suggest_categorical("bias", search_space["bias"]),
    }

    # 去重逻辑（组合转为不可变的 tuple）
    key = tuple(sorted(params.items()))
    if key in used_combinations:
        raise optuna.TrialPruned()  # 跳过重复组合
    used_combinations.add(key)

    # 构建命令
    cmd = [
        "python3",
        "training.py",
        "--dataset",
        dataset,
        "--source",
        source,
        "--epochs",
        "1000",
        "--seed",
        "42",
        "--hid",
        str(params['hid']),
        "--nlayers",
        str(params['nlayers']),
        "--K",
        "10",
        "--patience",
        "200",
        "--lr1",
        str(params['lr1']),
        "--lr2",
        str(params['lr2']),
        "--wd1",
        str(params['wd1']),
        "--wd2",
        str(params['wd2']),
        "--dpC",
        str(params['dpC']),
        "--dpM",
        str(params['dpM']),
        "--tau",
        str(params['tau']),
        "--dropout",
        str(params['dropout']),
        "--dev",
        gpu,
        "--bias",
        str(params['bias']),
    ]

    # 执行命令并捕获输出
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # 保存日志到 logs/{dataset}_{trial_id}.log
    log_path = Path(f"logs/{dataset}/")
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path.joinpath(f'{trial_id}.log'), "w", encoding='utf-8') as f:
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
study.optimize(objective, n_trials=128, n_jobs=6)
# squirrel ch 256 15
# blog flickr 128 6

print("Best trial:", study.best_trial)
print(" Value (accuracy):", -study.best_value)
print(" Params:", study.best_params)
