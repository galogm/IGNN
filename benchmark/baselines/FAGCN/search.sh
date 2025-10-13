id=0;metric=acc;d=flickr;gpu=$id; nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > logs/$d-$id.log &

id=0;metric=acc;d=blogcatalog;gpu=$id; nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > logs/$d-$id.log &

id=0;metric=acc;d=actor;gpu=$id; nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > logs/$d-$id.log &

id=0;metric=acc;d=chameleon;gpu=$id; nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > logs/$d-$id.log &

id=0;metric=acc;d=squirrel;gpu=$id; nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > logs/$d-$id.log &

id=0;metric=acc;d=amazon-ratings;gpu=$id; nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > logs/$d-$id.log &

id=0;metric=acc;d=roman-empire;gpu=$id; nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > logs/$d-$id.log &

id=0;metric=acc;d=photo;gpu=$id; nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > logs/$d-$id.log &

id=0;metric=acc;d=pubmed;gpu=$id; nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > logs/$d-$id.log &


id=0;metric=acc;d=wikics;gpu=$id; nohup python -u -m search --gpu=$gpu --metric=$metric --dataset=$d --n_trials=64 --n_jobs=1 > logs/$d-$id.log &
