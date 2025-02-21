```python
search_space_OrderedGNN = {
    'model_type': {'_type': 'choice', '_value': ['OrderedGNN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.001, 0.005, 0.01]},
    'l2_coef': {'_type': 'choice', '_value': [0.00000005, 0.000005, 0.0005, 0.05]},
    'dropout': {'_type': 'quniform', '_value': [0, 0.4, 0.1]},
    'dropout2': {'_type': 'quniform', '_value': [0, 0.4, 0.1]},
    'global_gating':  {'_type': 'cho ice', '_value': [True, False]},
    'num_layers_input':  {'_type': 'choice', '_value': [1, 2, 3]},
    'num_layers':  {'_type': 'choice', '_value': [1, 2, 3, 4, 8,10, 16,32, 64]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_GPRGNN = {
    'model_type': {'_type': 'choice', '_value': ['GPRGNN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.002, 0.05, 0.01]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.0005]},
    'alpha':  {'_type': 'choice', '_value': [0.1, 0.2, 0.5, 0.9]},
    'nlayers': {'_type': 'choice', '_value': [1, 2, 3, 4, 8,10, 16, 32]},
    'dprate': {'_type': 'choice', '_value': [0, 0.5, 0.7]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_GCNII = {
    'model_type': {'_type': 'choice', '_value': ['GCNII']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.005, 0.01, 0.05]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00005, 0.0001, 0.0005]},
    'dropout': {'_type': 'choice', '_value': [0, 0.5]},
    'alpha':  {'_type': 'quniform', '_value': [0.1, 0.5, 0.1]},
    'lammbda':  {'_type': 'choice', '_value': [0.5, 1.0, 1.5]},
    'layers':  {'_type': 'choice', '_value': [1, 2, 3, 4, 8, 10,16, 32, 64]},
    'variant':  {'_type': 'choice', '_value': [True, False]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_GGCN = {
    'model_type': {'_type': 'choice', '_value': ['GGCN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.01]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]},
    'dropout': {'_type': 'quniform', '_value': [0, 0.7, 0.1]},
    'decay_rate':  {'_type': 'quniform', '_value': [0., 1.5, 0.1]},
    'nhidden':  {'_type': 'choice', '_value': [8, 16, 32, 64, 80]},
    'nlayers':  {'_type': 'choice', '_value': [1, 2, 3, 4, 8, 16, 32]},
    'use_sparse':  {'_type': 'choice', '_value': [True]},
    # 'use_sparse':  {'_type': 'choice', '_value': [False]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_APPNP = {
    'model_type': {'_type': 'choice', '_value': ['APPNP']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.001, 0.005, 0.01, 0.05]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.0005]},
    'alpha':  {'_type': 'choice', '_value': [0.1, 0.2, 0.5]},
    'iterations': {'_type': 'choice', '_value': [1, 2, 3, 4, 8, 10]},
    'patience':  {'_type': 'choice', '_value': [100, 200]},
}
search_space_GloGNN = {
    'model_type': {'_type': 'choice', '_value': ['GloGNN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.01, 0.005]},
    'l2_coef': {'_type': 'choice', '_value': [0.00001, 0.00005, 0.0001]},
    'dropout': {'_type': 'quniform', '_value': [0, 0.9, 0.1]},
    'alpha':  {'_type': 'choice', '_value': [0, 1, 10]},
    'beta':  {'_type': 'choice', '_value': [0.1, 1, 10, 100, 1000]},
    'gamma':  {'_type': 'quniform', '_value': [0, 0.9, 0.1]},
    'delta':  {'_type': 'quniform', '_value': [0, 1, 0.1]},
    'norm_layers':  {'_type': 'choice', '_value': [1, 2, 3]},
    'orders':  {'_type': 'quniform', '_value': [1,  2, 3, 4,5,8, 10]},
    'norm_func_id':  {'_type': 'choice', '_value': [1, 2]},
    'early_stopping':  {'_type': 'choice', '_value': ￼},
}
search_space_MLP = {
    'model_type': {'_type': 'choice', '_value': ['MLP']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'normalize': {'_type': 'choice', '_value': [-1, 1]},
    'lr': {'_type': 'choice', '_value': [0.001, 0.005, 0.01, 0.05]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00005, 0.0001, 0.0005, 0.001]},
    'layers': {'_type': 'choice', '_value': [1, 2, 3]},
    'patience':  {'_type': 'choice', '_value': [100, 200]},
    'dropout': {'_type': 'choice', '_value': [0, 0.5]},
    'nhidden': {'_type': 'choice', '_value': [16, 32, 64]},
}
search_space_GCN = {
    'model_type': {'_type': 'choice', '_value': ['GCN']},
    'layers': {'_type': 'choice', '_value': [1, 2, 3, 4, 8, 10,16, 32, 64]},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'normalize': {'_type': 'choice', '_value': [-1, 1]},
    'lr': {'_type': 'choice', '_value': [0.001, 0.005, 0.01, 0.05]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00005, 0.0001, 0.0005, 0.001]},
    # 'layers': {'_type': 'choice', '_value': [1,]},
    'patience':  {'_type': 'choice', '_value': [100, 200]},
    'dropout': {'_type': 'choice', '_value': [0, 0.5]},
    'nhidden': {'_type': 'choice', '_value': [16, 32, 64]},
}
search_space_GAT = {
    'model_type': {'_type': 'choice', '_value': ['GAT']},
    'layers': {'_type': 'choice', '_value': [1, 2, 3, 4, 8, 10,16, 32, 64]},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'normalize': {'_type': 'choice', '_value': [-1, 1]},
    'lr': {'_type': 'choice', '_value': [0.001, 0.005, 0.01, 0.05]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00005, 0.0001, 0.0005, 0.001]},
    # 'layers': {'_type': 'choice', '_value': [1,]},
    'patience':  {'_type': 'choice', '_value': [100, 200]},
    'dropout': {'_type': 'choice', '_value': [0, 0.5]},
    'nhidden': {'_type': 'choice', '_value': [16, 32, 64]},
}
search_space_SGC = {
    'model_type': {'_type': 'choice', '_value': ['SGC']},
    'layers': {'_type': 'choice', '_value': [1, 2, 3, 4, 8, 10,16, 32, 64]},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'normalize': {'_type': 'choice', '_value': [-1, 1]},
    'lr': {'_type': 'choice', '_value': [0.001, 0.005, 0.01, 0.05]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00005, 0.0001, 0.0005, 0.001]},
    # 'layers': {'_type': 'choice', '_value': [1,]},
    'patience':  {'_type': 'choice', '_value': [100, 200]},
    'dropout': {'_type': 'choice', '_value': [0, 0.5]},
    'nhidden': {'_type': 'choice', '_value': [16, 32, 64]},
}
search_space_GraphSAGE = {
    'model_type': {'_type': 'choice', '_value': ['GraphSAGE']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'layers': {'_type': 'choice', '_value': [1, 2, 3, 4, 8, 10,16, 32, 64]},
    'normalize': {'_type': 'choice', '_value': [-1, 1]},
    'lr': {'_type': 'choice', '_value': [0.001, 0.005, 0.01, 0.05]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00005, 0.0001, 0.0005, 0.001]},
    # 'layers': {'_type': 'choice', '_value': [1,]},
    'patience':  {'_type': 'choice', '_value': [100, 200]},
    'dropout': {'_type': 'choice', '_value': [0, 0.5]},
    'nhidden': {'_type': 'choice', '_value': [16, 32, 64]},
}
search_space_MixHop = {
    'model_type': {'_type': 'choice', '_value': ['MixHop']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.001, 0.005, 0.01]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_H2GCN = {
    'model_type': {'_type': 'choice', '_value': ['H2GCN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.001, 0.005, 0.01]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]},
    'dropout': {'_type': 'choice', '_value': [0, 0.5]},
    'use_relu': {'_type': 'choice', '_value': [True, False]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_HOGGCN = {
    'model_type': {'_type': 'choice', '_value': ['HOGGCN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.001, 0.005, 0.01]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_GBKGNN = {
    'model_type': {'_type': 'choice', '_value': ['GBKGNN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.001, 0.0001, 0.00001]},
    'l2_coef': {'_type': 'choice', '_value': [0.01, 0.001, 0.0001]},
    'lamda': {'_type': 'quniform', '_value': [1, 64, 1]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_ACMGNN = {
    'model_type': {'_type': 'choice', '_value': ['ACMGNN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.01, 0.05]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]},
    'dropout': {'_type': 'quniform', '_value': [0, 0.9, 0.1]},
    'structure_info':  {'_type': 'choice', '_value': [0, 1]},
    # 'structure_info':  {'_type': 'choice', '_value': [0]},
    'variant':  {'_type': 'choice', '_value': [0, 1]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_JKNet = {
    'model_type': {'_type': 'choice', '_value': ['JKNet']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.01, 0.001]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]},
    'dropout': {'_type': 'quniform', '_value': [0.2, 0.5, 0.8]},
    'n_layers':  {'_type': 'choice', '_value': [1, 2, 3, 4, 8]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_SIGN = {
    'model_type': {'_type': 'choice', '_value': ['SIGN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.01, 0.001]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]},
    'dropout': {'_type': 'quniform', '_value': [0.2, 0.5, 0.8]},
    'n_hops':  {'_type': 'choice', '_value': [1, 2, 3, 4, 8,10, 16, 32, 64]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_IncepGCN = {
    'model_type': {'_type': 'choice', '_value': ['IncepGCN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.01, 0.001]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]},
    'dropout': {'_type': 'quniform', '_value': [0.2, 0.5, 0.8]},\
    'n_layers':  {'_type': 'choice', '_value': [1, 2, 3, 4,8]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_DAGNN = {
    'model_type': {'_type': 'choice', '_value': ['DAGNN']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.01, 0.001]},
    'l2_coef': {'_type': 'choice', '_value': [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]},
    'dropout': {'_type': 'quniform', '_value': [0.2, 0.5, 0.8]},
    'k':  {'_type': 'choice', '_value': [1, 2, 3, 4, 8,10, 16, 32, 64]},
    'n_hidden':  {'_type': 'choice', '_value': [64,128,256,512]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_NodeFormer = {
    'model_type': {'_type': 'choice', '_value': ['NodeFormer']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.01, 0.001]},
    'dropout': {'_type': 'quniform', '_value': [0.2, 0.5, 0.8]},
    'nhidden':  {'_type': 'choice', '_value': [128]},
    'num_layers':  {'_type': 'choice', '_value': [1,2,4,8]},
    'lambda':  {'_type': 'choice', '_value':[0, 0.1, 1] ￼},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_DIFFormer = {
    'model_type': {'_type': 'choice', '_value': ['DIFFormer']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.01, 0.001]},
    'dropout': {'_type': 'quniform', '_value': [0.2, 0.5, 0.8]},
    'nhidden':  {'_type': 'choice', '_value': [32,64,128]},
    'num_layers':  {'_type': 'choice', '_value': [1,2,4,8]},
    'patience':  {'_type': 'choice', '_value': ￼},
}
search_space_SGFormer = {
    'model_type': {'_type': 'choice', '_value': ['SGFormer']},
    'dataset': {'_type': 'choice', '_value': dataset_list},
    'lr': {'_type': 'choice', '_value': [0.01, 0.001]},
    'dropout': {'_type': 'quniform', '_value': [0.2, 0.5, 0.8]},
    'nhidden':  {'_type': 'choice', '_value': [32,64,128]},
    'num_layers':  {'_type': 'choice', '_value': [1,2,4,8]},
    'graph_weight':  {'_type': 'choice', '_value': [0.2, 0.5, 0.8]},
    'patience':  {'_type': 'choice', '_value': ￼},
}

```
