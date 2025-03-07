RN=concat
log_path=logs/$RN
mkdir -p $log_path
mkdir -p $log_path/large

v=1.0
IN=gcn-IN-SN

d=actor s=pyg
f=512
g=0
hops=1
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -n ln >$log_path/$d.log &

d=chameleon s=critical
f=512
g=0
hops=64
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -n False >$log_path/$d.log &

d=squirrel s=critical
f=512
g=0
hops=64
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops >$log_path/$d.log &

d=flickr s=cola
f=512
g=0
hops=10
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -n False >$log_path/$d.log &

d=blogcatalog s=cola
f=512
g=0
hops=10
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops >$log_path/$d.log &

d=roman-empire s=critical
f=300
g=0
hops=1
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops >$log_path/$d.log &

d=amazon-ratings s=critical
f=300
g=0
hops=16
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops >$log_path/$d.log &

d=photo s=pyg
f=256
g=0
hops=16
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops >$log_path/$d.log &

d=pubmed s=pyg
f=500
g=0
hops=8
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops >$log_path/$d.log &

d=wikics s=pyg
f=300
g=0
hops=8
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops >$log_path/$d.log &

d=arxiv s=ogb
g=0
f=512
hops=10
layers=1
lr=0.001
l2_coef=0.00005
nas_dropout=0
nss_dropout=0.8
clf_dropout=0.5
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -nas_dropout $nas_dropout -nss_dropout $nss_dropout -clf_dropout $clf_dropout -n ln >$log_path/large/$g-$d-$hops-$layers-$f-$lr-$l2_coef-$nas_dropout-$nss_dropout-$clf_dropout.log &

d=products s=ogb
g=0
f=200
hops=5
layers=1
lr=0.001
l2_coef=0.00000
nas_dropout=0
nss_dropout=0.5
clf_dropout=0.5
m=IGNN-$IN-$RN
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -nas_dropout $nas_dropout -nss_dropout $nss_dropout -clf_dropout $clf_dropout --eval_start 100 -i 1 -n ln >$log_path/large/$g-$d-$hops-$layers-$f-$lr-$l2_coef-$nas_dropout-$nss_dropout-$clf_dropout-$b.log &

d=pokec s=linkx
g=0
f=256
hops=6
layers=1
lr=0.001
l2_coef=0.0000005
nas_dropout=0
nss_dropout=0.2
clf_dropout=0.2
m=IGNN-$IN-$RN
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21  nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -nas_dropout $nas_dropout -nss_dropout $nss_dropout -clf_dropout $clf_dropout -eval 1200 -i 1 -n bn >$log_path/large/$g-$d-$hops-$layers-$f-$lr-$l2_coef-$nas_dropout-$nss_dropout-$clf_dropout-$b.log &
