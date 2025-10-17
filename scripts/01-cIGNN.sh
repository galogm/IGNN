v=1.0
agg=gcn_incep
IN=IN-SN
RN=concat
fs=True
log_path=logs/$RN
mkdir -p $log_path
mkdir -p $log_path/large

d=actor s=pyg
f=512
g=5
hops=1
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -n ln -fs $fs -a $agg >$log_path/$d-$m-$fs.log 2>&1 & echo "Check logs in $log_path/$d-$m-$fs.log. PID: $! "

d=chameleon s=critical
f=512
g=5
hops=64
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -n none -fs $fs -a $agg >$log_path/$d-$m-$fs.log 2>&1 & echo "Check logs in $log_path/$d-$m-$fs.log. PID: $! "

d=squirrel s=critical
f=512
g=5
hops=64
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -fs $fs -a $agg >$log_path/$d-$m-$fs.log & echo "Check logs in $log_path/$d-$m-$fs.log. PID: $! "

d=flickr
s=cola
f=512
g=5
hops=10
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -n none -fs $fs -a $agg >$log_path/$d-$m-$fs.log & echo "Check logs in $log_path/$d-$m-$fs.log. PID: $! "

d=blogcatalog s=cola
f=512
g=4
hops=10
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -fs $fs -a $agg >$log_path/$d-$m-$fs.log & echo "Check logs in $log_path/$d-$m-$fs.log. PID: $! "

d=roman-empire s=critical
f=300
g=3
hops=1
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -fs $fs  -a $agg >$log_path/$d-$m-$fs.log & echo "Check logs in $log_path/$d-$m-$fs.log. PID: $! "

d=amazon-ratings s=critical
f=300
g=1
hops=16
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -fs $fs -a $agg >$log_path/$d-$m-$fs.log & echo "Check logs in $log_path/$d-$m-$fs.log. PID: $! "

d=photo s=pyg
f=256
g=5
hops=16
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -fs $fs  -a $agg >$log_path/$d-$m-$fs.log & echo "Check logs in $log_path/$d-$m-$fs.log. PID: $! "

d=pubmed s=pyg
f=500
g=3
hops=3
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -fs $fs -a $agg >$log_path/$d-$m-3pre.log & echo "Check logs in $log_path/$d-$m-$fs.log. PID: $! "

d=wikics s=pyg
f=300
g=2
hops=8
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -fs $fs  -a $agg >$log_path/$d-$m-$fs.log & echo "Check logs in $log_path/$d-$m-$fs.log. PID: $! "

d=arxiv s=ogb
g=1
f=512
hops=10
layers=1
lr=0.001
l2_coef=0.00005
pre_dropout=0
hid_dropout=0.8
clf_dropout=0.5
m=IGNN-$IN-$RN
lp=$log_path/large/$g-$d-$hops-$layers-$f-$lr-$l2_coef-$pre_dropout-$hid_dropout-$clf_dropout.log
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -pre_dropout $pre_dropout -hid_dropout $hid_dropout -clf_dropout $clf_dropout -n ln -fs False -a $agg > $lp & echo "Check logs in $lp . PID: $! "

d=products s=ogb
g=1
f=200
hops=5
layers=1
lr=0.003
l2_coef=0.00000
pre_dropout=0
hid_dropout=0.5
clf_dropout=0.5
m=IGNN-$IN-$RN
lp=$log_path/large/$g-$d-$hops-$layers-$f-$lr-$l2_coef-$pre_dropout-$hid_dropout-$clf_dropout.log
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -pre_dropout $pre_dropout -hid_dropout $hid_dropout -clf_dropout $clf_dropout -p 10 --eval_start 150 -i 3 -n ln -fs False -pre True -a $agg > $lp & echo "Check logs in $lp . PID: $! "

d=pokec s=linkx
g=4
f=256
hops=6
layers=1
lr=0.001
l2_coef=0.0000005
pre_dropout=0
hid_dropout=0.2
clf_dropout=0.2
m=IGNN-$IN-$RN
lp=$log_path/large/$g-$d-$hops-$layers-$f-$lr-$l2_coef-$pre_dropout-$hid_dropout-$clf_dropout.log
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21  nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -pre_dropout $pre_dropout -hid_dropout $hid_dropout -clf_dropout $clf_dropout -p 10 -eval 1200 -i 3 -n bn -fs False -pre True -a $agg > $lp & echo "Check logs in $lp . PID: $! "
