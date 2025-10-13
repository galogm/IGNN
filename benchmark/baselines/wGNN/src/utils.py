import random
import string

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate


def print_files(files_to_print):
    for filename in files_to_print:
        f = open(filename, "r")
        print("\n" * 3)
        print(f"file version of {filename}")
        print("=" * 80)
        for line in f:
            print(line, end="", flush=True)
        print("=" * 80)
        print("\n" * 3)

    print("\n" * 5)
    print("=" * 80, flush=True)


def print_settings(args):
    # log device
    try:
        print("Device List: ", os.environ["CUDA_VISIBLE_DEVICES"])
    except:
        print("Using default device")

    # log all parameters
    for key, value in args.__dict__.items():
        print(f"{key}: \t {value}")


from sklearn.model_selection import StratifiedKFold


def separate_data(graph_indices, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    idx_list = []
    for idx in skf.split(np.zeros(len(graph_indices)), graph_indices):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    return train_idx, test_idx
