#  IGNN

<div align="center">

</div>

NeurIPS 25 Poster: [Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness–Generalization Perspective](https://neurips.cc/virtual/2025/poster/118841).

## Installation

**Requirements:**

* Python ≥ 3.8
* See venv installation scripts: [`./.ci/install-dev.sh`](./.ci/install-dev.sh), [`./.ci/install.sh`](./.ci/install.sh)
* Dependencies: [`./requirements-dev.txt`](./requirements-dev.txt), [`./requirements.txt`](./requirements.txt).

**Install scripts:**

```bash
# Development environment
$ bash .ci/install-dev.sh

# Production environment
$ bash .ci/install.sh
```

## Usage

Running scripts for each IGNN variant across all datasets:

| Variant | Script                                           |
| ------- | ------------------------------------------------ |
| c-IGNN  | [`./scripts/01-cIGNN.sh`](./scripts/01-cIGNN.sh) |
| r-IGNN  | [`./scripts/02-rIGNN.sh`](./scripts/02-rIGNN.sh) |
| a-IGNN  | [`./scripts/03-aIGNN.sh`](./scripts/03-aIGNN.sh) |

Hyperparameter searching scripts for each IGNN variant across all datasets:

| Variant | Script                                                   |
| ------- | -------------------------------------------------------- |
| c-IGNN  | [`./scripts/cignn_search.py`](./scripts/cignn_search.py) |
| r-IGNN  | [`./scripts/rignn_search.py`](./scripts/rignn_search.py) |
| a-IGNN  | [`./scripts/aignn_search.py`](./scripts/aignn_search.py) |

## Datasets and Splits

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

To reduce variance across datasets caused by heterogeneous splitting policies, we use a **unified 10× random split scheme** with a **48%/32%/20%** train/validation/test ratio.

- For medium-size datasets, the splits are stored in [`./data/random_splits/fixed_splits/`](./data/random_splits/fixed_splits/) and can be loaded as follows:

```python
from utils import get_splits

train_mask, val_mask, test_mask = get_splits(
    data=data,
    name=data.name,
    n_nodes=data.num_nodes,
    i=10,
    TRAIN_RATIO=48,
    VALID_RATIO=32,
    DATA=DATA_INFO
)
```

- For large datasets, `OGB-arxiv` and `OGB_products` are using the public splits with `pokec` using the splits from [this previous work](https://github.com/cornell-zhang/Polynormer).


## Baselines

The code for all 30 baselines is in [`./semi_heter/baselines/`](./semi_heter/baselines/):

* If a baseline has its **own folder**, a `search.py` script is included for hyperparameter tuning with `optuna`. See the `README.md` in the folder for details.
* If a baseline does **not** have its own folder, it can be run with a script like [`./baselines.py`](./baselines.py), which can conveniently derive the corresponding `search.py` script.
* All search spaces used in the experiments are documented in [`./configs/search_grid.py`](./configs/search_grid.py).

## Citation

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
