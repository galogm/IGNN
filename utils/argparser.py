"""Parse All Model Args"""

import argparse
import os
from collections import defaultdict
from pathlib import Path, PurePath
from typing import Dict

import yaml
from the_utils import tab_printer

type_map = {"int": int, "str": str, "float": float, "bool": bool}

#: Info of the models supported.
models: Dict = {
    "pca_kmeans": {
        "name": "PCA",
        "description": "PCA with Kmeans.",
        "paper url": "",
        "source code": "",
    },
}


def get_pkg_dir():
    return os.path.abspath(
        f"{os.path.dirname(os.path.realpath(__file__))}/..",
    )


def set_args_wrt_dataset(args, args_dict):
    """load experiment's changed model hyparamenter from yaml file

    Parameters
    ----------
    args : ArgumentParser
        ArgumentParser
    args_dict : dict
        dict of arguments from hyparamenter yaml file
    """
    # with open(args_dict["config"], "r", encoding="utf-8") as f:
    #     cfg_file = f.read()
    cfg_dict = read_configs(args_dict["model"])[args_dict["dataset"]]
    for k, v in cfg_dict.items():
        setattr(args, k, v)
    return args


def read_configs(_type: str = None) -> Dict:
    pkg_dir = get_pkg_dir()
    config_path: PurePath = Path(f"{pkg_dir}/configs/{_type}.yaml")
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        args = yaml.safe_load(f)
    return args


def set_subparser(model: str, _parser: argparse.ArgumentParser) -> None:
    args = read_configs(model)
    for key, val in args.items():
        keys = val.keys()
        default_val = val["default"] if "default" in keys else None
        type_val = type_map[val["type"]] if "type" in keys else type(val["default"])
        nargs_val = val["nargs"] if "nargs" in keys else None
        _parser.add_argument(
            f"--{key}",
            type=type_val,
            default=default_val,
            help=val["help"],
            nargs=nargs_val,
        )


def parse_all_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="HeteroGNN",
        description="Parameters for Heterophilous GNNs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Dataset used in the experiment",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="ID(s) of gpu used by cuda",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./data",
        help="Path to store the dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4096,
        help="Random seed. Defaults to 4096.",
    )

    subparsers = parser.add_subparsers(dest="model", help="sub-command help")

    for _model, items in models.items():
        # pylint: disable=invalid-sequence-index
        _help = items["description"]
        _parser = subparsers.add_parser(
            _model,
            help=f"Run Graph Clustering on {_help}",
        )
        set_subparser(_model, _parser)

    args = parser.parse_args()

    tab_printer(args)

    return args


def get_default_args(model: str) -> Dict:
    """Get default args of any model supported.

    Args:
        model (str): name of the model.

    Returns:
        Dict: the default args of the model.
    """
    _args = read_configs(model)
    args = {}
    for key, val in _args.items():
        keys = val.keys()
        default_val = val["default"] if "default" in keys else None
        args[key] = default_val

    return args


def parse_ignn_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="IGNN",
        description="",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=int,
        default=0,
        help="gpu id",
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=float,
        default=1,
        help="version",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="actor",
        help="dataset",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default="dgl",
        help="source",
    )
    parser.add_argument(
        "-r",
        "--return_type",
        type=str,
        default="dgl",
        help="return_type",
    )
    parser.add_argument(
        "-p",
        "--num_parts",
        type=int,
        default=10,
        help="Graph partition parts for batch learning",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="ignn",
        help="model",
    )
    parser.add_argument(
        "-fs",
        "--fast",
        type=lambda x: defaultdict(
            lambda: None,
            {
                "None": None,
                "True": True,
                "False": False,
            },
        )[x],
        choices=[True, False, None],
        default=None,
        help="fast version",
    )
    parser.add_argument(
        "-a",
        "--agg_type",
        choices=["gcn", "gcn_incep", "sage", "gat"],
        default="gcn",
        help="AGG type",
    )
    parser.add_argument(
        "-IN",
        "--IN",
        type=str,
        choices=["IN-SN", "IN-nSN", "nIN-nSN"],
        default="IN-SN",
        help="IN",
    )
    parser.add_argument(
        "-RN",
        "--RN",
        type=str,
        choices=["none", "concat", "attentive", "residual", None],
        default=None,
        help="RN",
    )
    parser.add_argument(
        "-f",
        "--h_feats",
        type=int,
        default=None,
        help="h feats",
    )
    parser.add_argument(
        "-pre_dropout",
        "--pre_dropout",
        type=lambda x: None if x == "None" else float(x),
        default="None",
        help="pre_dropout",
    )
    parser.add_argument(
        "-hid_dropout",
        "--hid_dropout",
        type=lambda x: None if x == "None" else float(x),
        default="None",
        help="hid_dropout",
    )
    parser.add_argument(
        "-clf_dropout",
        "--clf_dropout",
        type=lambda x: None if x == "None" else float(x),
        default="None",
        help="clf_dropout",
    )
    parser.add_argument(
        "-n",
        "--norm_type",
        choices=["bn", "ln", "none", None],
        default=None,
        help="Normalization type: 'bn' (BatchNorm) or 'ln' (LayerNorm) or 'none' (None)",
    )
    parser.add_argument(
        "-ac",
        "--act_type",
        type=str,
        choices=["relu", "prelu", None, "none"],
        default=None,
        help="act type",
    )
    parser.add_argument(
        "-at",
        "--att_act_type",
        choices=["tanh", "sigmoid", "softmax", "relu", "prelu", "leakyrelu", "gelu", "none", None],
        type=str,
        default=None,
        help="act attentive",
    )
    parser.add_argument(
        "-lr",
        "--lr",
        type=lambda x: None if x == "None" else float(x),
        default="None",
        help="lr",
    )
    parser.add_argument(
        "-l2_coef",
        "--l2_coef",
        type=lambda x: None if x == "None" else float(x),
        default="None",
        help="l2_coef",
    )
    parser.add_argument(
        "-hops",
        "--n_hops",
        type=lambda x: None if x == "None" else int(x),
        default="None",
        help="n_hops",
    )
    parser.add_argument(
        "-layers",
        "--n_layers",
        type=lambda x: None if x == "None" else int(x),
        default="None",
        help="n_layers",
    )
    parser.add_argument(
        "-pre",
        "--preln",
        type=lambda x: defaultdict(
            lambda: None,
            {
                "None": None,
                "True": True,
                "False": False,
            },
        )[x],
        choices=[True, False, None],
        default=None,
        help="pre linear transformation",
    )
    parser.add_argument(
        "-pub",
        "--public",
        type=lambda x: defaultdict(
            lambda: False,
            {
                "True": True,
                "False": False,
            },
        )[x],
        choices=[True, False],
        default=False,
        help="use public splits",
    )
    parser.add_argument(
        "-ne",
        "--n_epochs",
        type=int,
        default=3000,
        help="maximum epochs",
    )
    parser.add_argument(
        "-rp",
        "--repeat",
        type=int,
        default=None,
        help="repeats",
    )
    parser.add_argument(
        "-es",
        "--early_stop",
        type=lambda x: None if x == "None" else int(x),
        default="None",
        help="early stop",
    )
    parser.add_argument(
        "-eval",
        "--eval_start",
        type=int,
        default=0,
        help="eval_start epoch",
    )
    parser.add_argument(
        "-i",
        "--eval_interval",
        type=int,
        default=1,
        help="eval interval",
    )
    return parser.parse_args()
