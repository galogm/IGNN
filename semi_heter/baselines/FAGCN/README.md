Forked from https://github.com/bdy9527/FAGCN .

Parameter Search:
```bash
id=0;metric=acc;d=chameleon;gpu=$id;path=logs/$d;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=3 > $path/$metric-$id.log &
```


# FAGCN
Code of [Beyond Low-frequency Information in Graph Convolutional Networks](http://shichuan.org/doc/102.pdf)

# Q&A
Any suggestion/question is welcome.

# Reference
If you make advantage of the FAGCN model in your research, please cite the following in your manuscript:

```
@inproceedings{fagcn2021,
  title={Beyond Low-frequency Information in Graph Convolutional Networks},
  author={Deyu Bo and Xiao Wang and Chuan Shi and Huawei Shen},
  booktitle = {{AAAI}},
  publisher = {{AAAI} Press},
  year      = {2021}
}
```
