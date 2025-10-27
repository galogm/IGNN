Downloaded from its Supplementary Material of ICLR 2023 openreview: https://openreview.net/forum?id=fwn2Mqpy4pS , although it was accepted by ICML 2023.

Searching scripts:

```bash
id=5;metric=acc;d=squirrel;gpu=$id;path=logs/$d;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=32 --n_jobs=2 > $path/$metric-$id.log &
```
