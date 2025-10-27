#  IGNN

<div align="center">

</div>

NeurIPS 25 Poster:

[Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness–Generalization Perspective](https://neurips.cc/virtual/2025/poster/118841).

## 🛠️ Installation

* Python ≥ 3.8.16
* See virtual environment setup scripts:
  * Development: [`./.ci/install-dev.sh`](./.ci/install-dev.sh)
  * Production: [`./.ci/install.sh`](./.ci/install.sh)
* Dependency lists:
  * [`./requirements-dev.txt`](./requirements-dev.txt)
  * [`./requirements.txt`](./requirements.txt)

## 🚀 Usage

Run each IGNN variant across all datasets using the following scripts on:
1. Public splits: [`./scripts/00-best-racIGNN-public.sh`](./scripts/00-best-racIGNN-public.sh)
2. Our splits:

| Variant | Script                                           |
| ------- | ------------------------------------------------ |
| c-IGNN  | [`./scripts/01-best-cIGNN.sh`](./scripts/01-best-cIGNN.sh) |
| r-IGNN  | [`./scripts/02-best-rIGNN.sh`](./scripts/02-best-rIGNN.sh) |
| a-IGNN  | [`./scripts/03-best-aIGNN.sh`](./scripts/03-best-aIGNN.sh) |

3. Results of the public and our splits are documentd in  [`./results/table_pub.csv`](./results/table_pub.csv) and [`./results/table_our.csv`](./results/table_our.csv).

> [!important]
> Experimental setups for our reported results:
> - [Setting 1] Tesla V100, with Python 3.9.15, PyTorch 2.0.1, and Cuda 11.7.
>
> We observed **performance discrepancies when using identical hyperparameters across different PyTorch/CUDA versions and GPU architectures, e.g., V100 (Setting 1) vs. RTX 3090 (Setting 2)**.
> - [Setting 2] RTX 3090, with Python 3.8.16, PyTorch 2.1.2, and Cuda 12.1.
>
> For instance, on Chameleon, the same config in [`./scripts/01-best-cIGNN.sh`](./scripts/01-best-cIGNN.sh) yielded `50.79 ± 4.92` (Setting 1) vs. `47.53 ± 3.36` (Setting 2).
> Although SOTA performance can be achieved under all environments with proper tuning, **optimal hyperparameters may differ across setups**.
>

Perform hyperparameter searches for each variant using:

| Split  | Search Script                                                                |
| ------ | ---------------------------------------------------------------------------- |
| Ours   | [`./scripts/00-search-ours-split.sh`](./scripts/00-search-ours-split.sh)     |
| Public | [`./scripts/00-search-public-split.sh`](./scripts/00-search-public-split.sh) |

| Variant | Script                                                       |
| ------- | ------------------------------------------------------------ |
| c-IGNN  | [`./scripts/01cignn_search.py`](./scripts/01cignn_search.py) |
| r-IGNN  | [`./scripts02rignn_search.py`](./scripts/02rignn_search.py)  |
| a-IGNN  | [`./scripts/03aignn_search.py`](./scripts/03aignn_search.py) |
> [!tip]
> We **strongly recommend performing your own hyperparameter search** to achieve the best performance in your environment using the above provided search scripts.

## 📊 Datasets and Splits

We use the open-source pip package [`graph_datasets`](https://github.com/galogm/graph_datasets) for unified data loading:

```bash
$ python -m pip install graph_datasets
```

**Example usage:**

```python
from graph_datasets import load_data
from configs import DataConf
from utils import read_configs

DATA_INFO = DataConf(**read_configs("data"))

data = load_data(
    dataset_name='squirrel',
    source='critical',
    directory=DATA_INFO.DATA_DIR,
    row_normalize=True,
    rm_self_loop=False,
    add_self_loop=True,
    verbosity=1,
    return_type="pyg",
)
```

To minimize variance from inconsistent split policies across datasets, we use a **unified 10× random split scheme** with a **48%/32%/20%** train/validation/test ratio.

- For medium-size datasets, the splits are stored in [`./data/random_splits/fixed_splits/`](./data/random_splits/fixed_splits/).
- For large datasets, `OGB-arxiv` and `OGB_products` are using the public splits with `pokec` using the splits from [this work](https://github.com/cornell-zhang/Polynormer).
- All splits can be loaded via:
```python
from utils import get_splits

# `repeat` is the number of our/public splits
# `i` is the index of the selected split
train_mask, val_mask, test_mask = get_splits(
    data=data,
    name=data.name,
    n_nodes=data.num_nodes,
    i=1,
    repeat=10,
    TRAIN_RATIO=48,
    VALID_RATIO=32,
    DATA=DATA_INFO,
    public=False,
)
```

## 🧩 Baselines

The code for all 30 baselines is in [`./benchmark/baselines`](./benchmark/baselines):

* If a baseline has its **own folder**, a `search.py` script is included for hyperparameter tuning with `optuna`. See the `README.md` in the folder for details.
* If a baseline does **not** have its own folder, it can be run with a script like [`./baselines.py`](./baselines.py), which can conveniently derive the corresponding `search.py` script.
* All search spaces used in the experiments are documented in [`./configs/search_grid.py`](./configs/search_grid.py).

## 📝 Empirical Analysis

* The code for the empirical analysis is documented in  [`./scripts/dml.sh`](./scripts/dml.sh).
* Run the analysis and draw the analysis plots via:
```bash
$ bash scripts/dml.sh squirrel critical False 10
$ python -u -m scripts.merge
```

## 📚 Citation

If you find this work useful, please cite our paper:
```kotlin
@inproceedings{ignn,
  title={Making Classic {GNN}s Strong Baselines Across Varying Homophily: A Smoothness{\textendash}Generalization Perspective},
  author={Ming Gu and Zhuonan Zheng and Sheng Zhou and Meihan Liu and Jiawei Chen and Qiaoyu Tan and Liangcheng Li and Jiajun Bu},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=IAGbhDARZd}
}
```
