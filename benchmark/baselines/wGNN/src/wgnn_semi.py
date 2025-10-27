import os
import sys

sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties("cuda:0"))
import argparse

from src import wgnn_graphops as GO
from src import wgnn_network as GN

parser = argparse.ArgumentParser(description="wgnn_semi_supervised")
parser.add_argument(
    "--dataset",
    default="Cora",
    type=str,
    help="dataset name",
)

parser.add_argument(
    "--omega",
    default=1,
    type=int,
    help="1 if use omegaGCN, 0 otherwise",
)

parser.add_argument(
    "--singleomega",
    default=0,
    type=int,
    help="1 if use a single omega (global), 0 otherwise",
)

parser.add_argument(
    "--attspat",
    default=0,
    type=int,
    help="1 if use attention for spatial operation, 0 otherwise",
)

parser.add_argument(
    "--attHeads",
    default=1,
    type=int,
    help="number of attention heads",
)

parser.add_argument(
    "--omegaPerChannel",
    default=0,
    type=int,
    help="omegaPerChannel to learn",
)

args = parser.parse_args()
nlayers = 2
ncheckpoints = 1
nomega = 1
omega_per_channel = 0
nheads = 1
args = parser.parse_args()
dataset = args.dataset
if dataset == "Cora":
    nNin = 1433
    n_channels = 64  # trial.suggest_categorical('n_channels', [64, 128, 256])
    dropout = 0.6

elif dataset == "CiteSeer":
    nNin = 3703
    n_channels = 256  # trial.suggest_categorical('n_channels', [64, 128, 256])
    dropout = 0.7

elif dataset == "PubMed":
    nNin = 500
    n_channels = 256  # trial.suggest_categorical('n_channels', [64, 128, 256])
    dropout = 0.5

nopen = n_channels
nhid = n_channels
n_layers = nlayers
path = "yourdatapath/" + dataset
transform = T.Compose([T.NormalizeFeatures()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

lr = 0.01
attLR = 0.01
lrGCN = 0.01
lrOmega = 0.01
wd = 1e-5
wdGCN = 1e-5
wdOmega = 1e-5
attWD = 1e-5

numAttHeads = args.attHeads if args.attspat else 1
model = GN.wgnn(
    nNin,
    nopen,
    nhid,
    nlayers,
    num_output=dataset.num_classes,
    dropOut=dropout,
    numAttHeads=numAttHeads,
    num_omega=nomega,
    omega_perchannel=nomega,
    singleOmega=args.singleomega,
)
model.reset_parameters()
model = model.to(device)
optimizer = torch.optim.Adam(
    [
        dict(params=model.KN1, lr=lrGCN, weight_decay=wdGCN),
        dict(params=model.K1Nopen, weight_decay=wd),
        dict(params=model.KNclose, weight_decay=wd),
        dict(params=model.att_src, lr=attLR, weight_decay=attWD),
        dict(params=model.att_dst, lr=attLR, weight_decay=attWD),
        dict(params=model.omega, lr=lrOmega, weight_decay=wdOmega),
    ],
    lr=lr,
)


def train():
    model.train()
    optimizer.zero_grad()
    I = data.edge_index[0, :]
    J = data.edge_index[1, :]
    N = data.y.shape[0]
    G = GO.graph(I, J, N, W=torch.ones_like(I).squeeze(), pos=None, faces=None)
    G = G.to(device)
    xn = data.x.t().unsqueeze(0)

    [out, _] = model(xn, G, omega=args.omega, attention=args.attspat, checkpoints=ncheckpoints)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    optimizer.step()
    return float(loss)


@torch.no_grad()
def eval_test():
    model.eval()
    I = data.edge_index[0, :]
    J = data.edge_index[1, :]
    N = data.y.shape[0]
    G = GO.graph(I, J, N, W=torch.ones_like(I).squeeze(), pos=None, faces=None)
    G = G.to(device)
    xn = data.x.t().unsqueeze(0)

    [out, _] = model(xn, G, omega=args.omega, attention=args.attspat, checkpoints=ncheckpoints)

    pred, accs = out.argmax(dim=-1), []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
patience = 0
for epoch in range(1, 10001):
    loss = train()
    train_acc, val_acc, tmp_test_acc = eval_test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        patience = 0
    patience += 1
    if patience > 400:
        break
    print(
        f"Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, "
        f"Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, "
        f"Final Test: {test_acc:.4f}",
        flush=True,
    )
