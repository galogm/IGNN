v=1.0
agg=gcn
IN=IN-nSN
RN=residual
log_path=logs/$RN
mkdir -p $log_path
mkdir -p $log_path/large

d=actor s=pyg
f=512
g=0
hops=2
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -a=$agg >$log_path/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=chameleon s=critical
f=512
g=1
hops=1
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -n none -IN $IN -RN $RN -hops $hops -a=$agg >$log_path/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=squirrel s=critical
f=512
g=0
hops=2
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -a=$agg >$log_path/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=flickr s=cola
f=512
g=2
hops=2
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -a=$agg >$log_path/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=blogcatalog s=cola
f=512
g=3
hops=4
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -a=$agg >$log_path/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=roman-empire s=critical
f=300
g=4
hops=2
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -a=$agg >$log_path/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=amazon-ratings s=critical
f=300
g=5
hops=2
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -a=$agg >$log_path/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=photo s=pyg
f=512
g=6
hops=4
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -a=$agg >$log_path/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=pubmed s=pyg
f=500
g=7
hops=2
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -a=$agg >$log_path/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=wikics s=pyg
f=300
g=1
hops=4
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -a=$agg >$log_path/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=arxiv s=ogb
g=1
f=512
hops=10
layers=1
lr=0.001
l2_coef=0.00005
nas_dropout=0
nss_dropout=0.8
clf_dropout=0.5
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -nas_dropout $nas_dropout -nss_dropout $nss_dropout -clf_dropout $clf_dropout -a=$agg >$log_path/large/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "
# $d-$hops-$layers-$f-$lr-$l2_coef-$nas_dropout-$nss_dropout-$clf_dropout.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=products s=ogb
g=2
f=200
hops=5
layers=1
lr=0.003
l2_coef=0.00000
nas_dropout=0
nss_dropout=0.5
clf_dropout=0.5
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -nas_dropout $nas_dropout -nss_dropout $nss_dropout -clf_dropout $clf_dropout --eval_start 100 -i 3 -n ln -p 5 -a=$agg >$log_path/large/$d-$hops-$layers-$f-$lr-$l2_coef-$nas_dropout-$nss_dropout-$clf_dropout-$b.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=pokec s=linkx
g=3
f=256
hops=5
layers=1
lr=0.001
l2_coef=0.000005
nas_dropout=0
nss_dropout=0.2
clf_dropout=0.2
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -nas_dropout $nas_dropout -nss_dropout $nss_dropout -clf_dropout $clf_dropout -eval 1200 -i 3  -n bn -a=$agg >$log_path/large/$d-$hops-$layers-$f-$lr-$l2_coef-$nas_dropout-$nss_dropout-$clf_dropout-$b.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "
