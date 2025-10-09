#  IGNN

<div align="center">

</div>

## Installation

- python>=3.8
- For installation scripts see [`.ci/install-dev.sh`](.ci/install-dev.sh), [`.ci/install.sh`](.ci/install.sh)
- For requirements, see [`requirements-dev.txt`](./requirements-dev.txt), [`requirements.txt`](./requirements.txt) or [`pyproject.toml:dependencies`](./pyproject.toml).

```bash
$ bash .ci/install-dev.sh
$ bash .ci/install.sh
```

## Usage

Detailed scripts for each IGNN variant across all datasets:

- c-IGNN: `scripts/01-cIGNN.sh`

- r-IGNN: `scripts/02-rIGNN.sh`

- a-IGNN: `scripts/03-aIGNN.sh`

## Datasets and Splits

We use a open-source pip package [`graph_datasets`](https://github.com/galogm/graph_datasets) for unified dataloaders of all datasets.

```bash
$ python -m pip install graph_datasets
```

```python
from graph_datasets import load_data
from ignn.modules import DataConf
from ignn.utils import read_configs

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

Guided by our theoretical emphasis on generalization, an aspect highly sensitive to dataset splits, we adopted a unified 10× random split scheme with a 48%/32%/20% train/validation/test ratio, which reduces variance across datasets stemming from heterogeneous splitting policies in earlier work.

The splits are in [`./data/random_splits/fixed_splits/`](./data/random_splits/fixed_splits/), and can be loaded with:

```python
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


## Baselines

The code of all 30 baselines can be found in the folder [`./semi_heter/baselines/`](./semi_heter/baselines/).
- If the baseline has a separate folder of its name, then a `search.py` script is placed in it, which can be used for hyper-parameter searching using `optuna`. Check the `README.md` in the folder for details.
- If a baseline has no folder of its name, then it can be run with [`./run_baselines.py`](./run_baselines.py).
- All searching spaces used in the experiments can be found in [`./configs/search_grid.md`](./configs/search_grid.md).
