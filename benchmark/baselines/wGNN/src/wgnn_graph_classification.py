import os
import os.path as osp
import sys

sys.path.append(os.getcwd())
import src.wgnn_graphops as GO
import src.wgnn_network as GN
import torch
import torch.nn.functional as F

num_workers = 6
gettrace = getattr(sys, "gettrace", None)
if gettrace is not None:
    num_workers = 0
###################################################################################
nlayers = 4
channels = 32
lr = 0.01
wd = 0
n_channels = 32
dropout = 0
bs = 32
attspat = 0  # 0 for omegaGCN, 1 for omegaGAT
nomega = 1  # 0 for omega per layer, 1 for per layer and per channel
if "mutag" in sys.argv:
    datasetstr = "MUTAG"
elif "proteins" in sys.argv:
    datasetstr = "PROTEINS"
elif "ptc" in sys.argv:
    datasetstr = "PTC_MR"
elif "nci1" in sys.argv:
    datasetstr = "NCI1"
elif "nci109" in sys.argv:
    datasetstr = "NCI109"
else:
    datasetstr = "MUTAG"

wdGCN = wd = 0
lrGCN = lr
import time
from datetime import datetime

start_time = time.time()
time_ = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from src.utils import print_files, separate_data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

data_path = osp.join("/yourdatapath/", "data", "TU", datasetstr)

dataset = TUDataset(data_path, name=datasetstr).shuffle()
seeds = np.arange(0, 1)
folds = np.arange(0, 10)
graph_indices = [data.y.item() for data in dataset]
results = torch.zeros((len(seeds), len(folds)))
for si, seed in enumerate(seeds):
    seed_results = []
    for fi in folds:
        train_idx, test_idx = separate_data(graph_indices, seed, fi)
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs)

        model = GN.wgnn(
            dataset.num_features,
            n_channels,
            n_channels,
            nlayers,
            num_output=dataset.num_classes,
            dropOut=dropout,
            numAttHeads=4 if attspat else 1,
            num_omega=1,
            TUDatasets=True,
            omega_perchannel=nomega,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        model.to(device)

        def train():
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                I = data.edge_index[0, :]
                J = data.edge_index[1, :]
                N = data.x.shape[0]

                G = GO.graph(I, J, N, pos=None, faces=None)
                G = G.to(device)
                output = model(
                    data.x.t().unsqueeze(0), G, data=data, omega=1, attention=attspat, checkpoints=1
                )
                loss = F.nll_loss(output, data.y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * data.num_graphs
            return total_loss / len(train_loader.dataset)

        @torch.no_grad()
        def valtest(loader):
            model.eval()

            total_correct = 0
            for data in loader:
                data = data.to(device)
                I = data.edge_index[0, :]
                J = data.edge_index[1, :]
                N = data.x.shape[0]

                G = GO.graph(I, J, N, pos=None, faces=None)
                G = G.to(device)
                output = model(
                    data.x.t().unsqueeze(0), G, data=data, omega=1, attention=attspat, checkpoints=1
                )
                total_correct += int((output.argmax(-1) == data.y).sum())
            return total_correct / len(loader.dataset)

        best_acc = 0
        for epoch in range(1, 301):
            loss = train()
            train_acc = valtest(train_loader)
            test_acc = valtest(test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
        print(
            f"Seed: {seeds[si]:03d}   Fold: {folds[fi]:03d}   Epoch: {epoch:03d},"
            f" Loss: {loss:.4f}, Train Acc: {train_acc:.4f} "
            f"Test Acc: {test_acc:.4f}, Best test Acc: {best_acc:.4f}",
            flush=True,
        )
        results[si, fi] = best_acc
        seed_results.append(best_acc)
    seed_results = np.array(seed_results)
    print(
        "Seed:",
        seeds[si],
        " avg test acc:",
        np.mean(seed_results),
        " std:",
        np.std(seed_results),
        flush=True,
    )

print("Results:", results, flush=True)
