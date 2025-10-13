Thank you for reviewing our paper.

In this readme file we explain on the attached code of our submission.

Datasets:
All datasets in our expeiments are freely and openly available, and are cited in the paper.
For convenience, we have attached Cora, Pubmed, Citeseer, Wisconsin, Texas, Cornell, Chameleon and Actor datasets in this submission as those are rather small datasets.

For the rest of the datasets, we use the dataloaders from the useful PyTorch-Geometric which implemented an automatic download of those datasets.

Files contents:
src/inits.py - initialization functions file
src/process.py - loader and data processing for full supervision experiments
src/utils.py - our utils functions.
src/utils_gcnii.py - a utils file based on the public repository of the authors of GCNII. used for the full supervision experiment to load the data.
src/wgnn_graphops.py - the graph operations functions file.
src/wgnn_network.py - the source code of our omegaGNN and in particular omegaGCN and omegaGAT.
src/wgnn_semi.py - the main file of the semi supervised node classification experiment.
src/wgnn_full.py - the main file of the full supervised node classification experiment.
src/wgnn_graph_classification.py - the main file of the graph classification experiment.

Experiments:
In order to run any of those experiments, simply type in the command line: python src/wgnn_semi.py, python src/wgnn_full.py or python src/wgnn_graph_classification.py ,
depending on the experiment you wish to run, according to the description above.
The expriments code includes both training and evaluation code.

Code dependencies:
We use the following python packages:
-pytorch
-numpy
-pytorch-geometric
-sklearn
-matplotlib
-scipy

We hope you find our code useful :)
