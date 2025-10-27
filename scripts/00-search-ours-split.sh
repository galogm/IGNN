# search on our unified 10x random splits
id=0;model="c-IGNN";d=chameleon;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.01cignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=1;model="c-IGNN";d=actor;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.01cignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=0;model="c-IGNN";d=squirrel;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.01cignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=0;model="c-IGNN";d=amazon-ratings;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.01cignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=1;model="c-IGNN";d=roman-empire;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.01cignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=1;model="c-IGNN";d=blogcatalog;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.01cignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=1;model="c-IGNN";d=flickr;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.01cignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=2;model="c-IGNN";d=pubmed;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.01cignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=2;model="c-IGNN";d=photo;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.01cignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=2;model="c-IGNN";d=wikics;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.01cignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!

id=3;model="r-IGNN";d=chameleon;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=4;model="r-IGNN";d=actor;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=3;model="r-IGNN";d=squirrel;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=3;model="r-IGNN";d=amazon-ratings;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=4;model="r-IGNN";d=roman-empire;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=4;model="r-IGNN";d=blogcatalog;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=4;model="r-IGNN";d=flickr;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=5;model="r-IGNN";d=pubmed;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=5;model="r-IGNN";d=photo;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=5;model="r-IGNN";d=wikics;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.02rignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!


id=6;model="a-IGNN";d=chameleon;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.03aignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=7;model="a-IGNN";d=actor;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.03aignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=6;model="a-IGNN";d=squirrel;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.03aignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=6;model="a-IGNN";d=amazon-ratings;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.03aignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=7;model="a-IGNN";d=roman-empire;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.03aignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=7;model="a-IGNN";d=blogcatalog;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.03aignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=7;model="a-IGNN";d=flickr;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.03aignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=2;model="a-IGNN";d=pubmed;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.03aignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=2;model="a-IGNN";d=photo;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.03aignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
id=5;model="a-IGNN";d=wikics;gpu=$id;log_path=logs/$model/$d;mkdir -p $log_path;nohup python -u -m scripts.03aignn_search --gpu=$gpu --dataset=$d --n_trials=256 --n_jobs=3 --repeat 10 > $log_path/$id-$d.log 2>&1 & echo $!
