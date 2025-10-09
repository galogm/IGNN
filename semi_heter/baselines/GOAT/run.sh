id=0;metric=acc;d=chameleon;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=2 > $path/$metric-$id.log &

id=0;metric=acc;d=squirrel;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=2 > $path/$metric-$id.log &

id=1;metric=acc;d=actor;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=2 > $path/$metric-$id.log &

id=2;metric=acc;d=flickr;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > $path/$metric-$id.log &

id=4;metric=acc;d=blogcatalog;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > $path/$metric-$id.log &

id=5;metric=acc;d=roman-empire;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > $path/$metric-$id.log &

id=6;metric=acc;d=amazon-ratings;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > $path/$metric-$id.log &

id=5;metric=acc;d=pubmed;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > $path/$metric-$id.log &

id=6;metric=acc;d=photo;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > $path/$metric-$id.log &

id=2;metric=acc;d=wikics;gpu=$id;path=logs/$d/$metric;mkdir -p $path && nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > $path/$metric-$id.log &
