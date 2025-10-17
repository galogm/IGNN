"""Default Params.
"""

from collections import defaultdict
from typing import Dict, Literal

pre_norms = defaultdict(
    lambda: True,
    {
        "chameleon": False,
        "squirrel": True,
        "actor": False,
        "flickr": True,
        "blogcatalog": True,
        "roman-empire": False,
        "amazon-ratings": True,
        "photo": True,
        "pubmed": False,
        "wikics": False,
        "arxiv": False,
        "products": False,
        "pokec": False,
    },
)

feats = defaultdict(
    lambda: 512,
    {
        "chameleon": 512,
        "squirrel": 512,
        "actor": 512,
        "photo": 256,
        "pubmed": 500,
        "roman-empire": 300,
        "amazon-ratings": 300,
        "flickr": 512,
        "blogcatalog": 512,
        "wikics": 300,
        "arxiv": 256,
        "products": 200,
        "pokec": 256,
    },
)

l2_coefs = defaultdict(
    lambda: 5e-5,
    {
        "chameleon": 5e-5,
        "squirrel": 5e-5,
        "actor": 0.0,
        "pubmed": 5e-5,
        "photo": 5e-8,
        "roman-empire": 5e-5,
        "amazon-ratings": 5e-5,
        "flickr": 5e-5,
        "blogcatalog": 5e-5,
        "wikics": 5e-5,
        "arxiv": 5e-5,
        "products": 5e-5,
        "pokec": 5e-8,
    },
)

lrs = defaultdict(
    lambda: 0.001,
    {
        "chameleon": 0.001,
        "squirrel": 0.001,
        "actor": 0.001,
        "pubmed": 0.001,
        "photo": 0.001,
        "roman-empire": 0.001,
        "amazon-ratings": 0.001,
        "flickr": 0.001,
        "blogcatalog": 0.001,
        "wikics": 0.001,
        "arxiv": 0.001,
        "products": 0.001,
        "pokec": 0.001,
    },
)

ess = defaultdict(
    lambda: 100,
    {
        "chameleon": 50,
        "squirrel": 100,
        "actor": 50,
        "pubmed": 100,
        "photo": 100,
        "roman-empire": 300,
        "amazon-ratings": 300,
        "flickr": 100,
        "blogcatalog": 100,
        "wikics": 200,
        "arxiv": 200,
        "products": 100,
        "pokec": 100,
    },
)

pre_dropouts = defaultdict(
    lambda: 0.5,
    {
        "flickr": 0.8,
        "blogcatalog": 0.8,
        "chameleon": 0.8,
        "squirrel": 0.8,
        "actor": 0.0,
        "pubmed": 0.5,
        "photo": 0.4,
        "roman-empire": 0.2,
        "amazon-ratings": 0.0,
        "wikics": 0.2,
        "arxiv": 0.0,
        "products": 0.0,
        "pokec": 0.0,
    },
)

hid_dropouts = defaultdict(
    lambda: 0.8,
    {
        "flickr": 0.8,
        "blogcatalog": 0.8,
        "chameleon": 0.8,
        "squirrel": 0.8,
        "actor": 0.8,
        "pubmed": 0.9,
        "photo": 0.8,
        "roman-empire": 0.2,
        "amazon-ratings": 0.8,
        "wikics": 0.5,
        "arxiv": 0.8,
        "products": 0.5,
        "pokec": 0.2,
    },
)

clf_dropouts = defaultdict(
    lambda: 0.5,
    {
        "flickr": 0.9,
        "blogcatalog": 0.9,
        "chameleon": 0.9,
        "squirrel": 0.9,
        "actor": 0.9,
        "photo": 0.5,
        "pubmed": 0.9,
        "roman-empire": 0.2,
        "amazon-ratings": 0.9,
        "wikics": 0.9,
        "arxiv": 0.5,
        "products": 0.5,
        "pokec": 0.2,
    },
)

n_hopss = defaultdict(
    lambda: 10,
    {
        "chameleon": 64,
        "squirrel": 64,
        "actor": 1,
        "flickr": 10,
        "blogcatalog": 10,
        "roman-empire": 1,
        "amazon-ratings": 16,
        "photo": 16,
        "pubmed": 4,
        "wikics": 8,
        "arxiv": 10,
        "products": 5,
        "pokec": 6,
    },
)

n_layerss = defaultdict(
    lambda: 1,
    {
        "chameleon": 1,
        "squirrel": 1,
        "actor": 1,
        "flickr": 1,
        "blogcatalog": 1,
        "roman-empire": 5,
        "amazon-ratings": 1,
        "photo": 1,
        "pubmed": 1,
        "wikics": 1,
        "arxiv": 1,
        "products": 1,
        "pokec": 1,
    },
)

RNs = defaultdict(
    lambda: "concat",
    {
        "chameleon": "concat",
        "squirrel": "concat",
        "actor": "concat",
        "flickr": "concat",
        "blogcatalog": "concat",
        "roman-empire": "concat",
        "amazon-ratings": "concat",
        "pubmed": "concat",
        "photo": "concat",
        "wikics": "concat",
        "arxiv": "concat",
        "products": "concat",
        "pokec": "concat",
    },
)

acts = defaultdict(
    lambda: "relu",
    {
        "chameleon": "relu",
        "squirrel": "relu",
        "actor": "relu",
        "flickr": "relu",
        "blogcatalog": "relu",
        "roman-empire": "relu",
        "amazon-ratings": "prelu",
        "photo": "relu",
        "pubmed": "relu",
        "wikics": "relu",
        "arxiv": "relu",
        "products": "relu",
        "pokec": "relu",
    },
)

norms: Dict[str, Literal["bn", "ln", "none"]] = defaultdict(
    lambda: "none",
    {
        "chameleon": "ln",
        "squirrel": "ln",
        "actor": "ln",
        "flickr": "ln",
        "blogcatalog": "ln",
        "roman-empire": "ln",
        "amazon-ratings": "ln",
        "photo": "ln",
        "pubmed": "ln",
        "wikics": "ln",
        "arxiv": "ln",
        "products": "ln",
        "pokec": "bn",
    },
)

att_norms: Dict[str, Literal["bn", "ln", "none"]] = defaultdict(
    lambda: "ln",
    {
        "chameleon": "none",
        "squirrel": "none",
        "actor": "ln",
        "flickr": "none",
        "blogcatalog": "ln",
        "roman-empire": "none",
        "amazon-ratings": "ln",
        "photo": "none",
        "pubmed": "ln",
        "wikics": "none",
        "arxiv": "none",
        "products": "none",
        "pokec": "bn",
    },
)

self_loop_attentive = defaultdict(
    lambda: True,
    {
        "chameleon": False,
        "squirrel": False,
        "actor": False,
        "flickr": False,
        "blogcatalog": False,
        "roman-empire": False,
        "amazon-ratings": False,
        "photo": True,
        "pubmed": False,
        "wikics": False,
        "arxiv": False,
        "products": False,
        "pokec": False,
    },
)

act_att = defaultdict(
    lambda: "tanh",
    {
        "chameleon": "tanh",
        "squirrel": "sigmoid",
        "actor": "tanh",
        "flickr": "tanh",
        "blogcatalog": "tanh",
        "roman-empire": "tanh",
        "amazon-ratings": "tanh",
        "photo": "tanh",
        "pubmed": "tanh",
        "wikics": "tanh",
        "arxiv": "tanh",
        "products": "tanh",
        "pokec": "tanh",
    },
)

repeats = defaultdict(
    lambda: 5,
    {
        "actor": 10,
        "blogcatalog": 10,
        "flickr": 10,
        "squirrel": 10,
        "chameleon": 10,
        "roman-empire": 10,
        "amazon-ratings": 10,
        "photo": 10,
        "pubmed": 10,
        "wikics": 10,
        "arxiv": 3,
        "products": 3,
        "pokec": 3,
    },
)

DATASETS = {
    "critical": [
        # 890
        "chameleon",
        # 2,223
        "squirrel",
        # # 10,000
        # "minesweeper",
        # # 11,758
        # "tolokers",
        # # 22,662
        # "roman-empire",
        # # 24,492
        # "amazon-ratings",
        # 48,921
        # "questions",
    ],
    "cola": [
        "flickr",
        "blogcatalog",
    ],
    "pyg": [
        # "texas",
        # "coRNell",
        # "wisconsin",
        # "corafull",
        # "cora",
        # "citeseer",
        "photo",
        "actor",
        "pubmed",
        "wikics",
    ],
    "Critical": [
        # 22,662
        "roman-empire",
        # 24,492
        "amazon-ratings",
        # 48,921
        # "questions",
    ],
    "ogb": [
        "arxiv",
        "proteins",
    ],
    "linkx": [
        # (array([0, 1, 2]), array([ 97, 504, 361]))
        # 962
        "Reed98",
        # # array([0, 1, 2]), array([ 418, 2153, 2609]
        # # 5,180
        "Johns Hopkins55",
        # # (array([0, 1, 2]), array([ 203, 1015, 1017]))
        # # 2,235
        "Amherst41",
        # # (array([0, 1, 2]), array([1838, 8135, 8687]))
        # # 18,660
        "CoRNell5",
        # # 41,554
        "Penn94",
        # # 168,114
        "twitch-gamers",
        # 169,343
        "arxiv-year",
        # 421,961
        "genius",
        # # 1,632,803
        "pokec",
        # 2,923,922
        "snap-patents",
        # 1,925,342
        "wiki",
    ],
}
