log_path=logs/concat/large
mkdir -p $log_path

v=1.0
nie=gcn-nie-nst

g=6
f=512
nrl=concat
hops=10
layers=1
lr=0.001
l2_coef=0.00005
nas_dropout=0
nss_dropout=0.8
clf_dropout=0.5
d=arxiv s=ogb
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -nas_dropout $nas_dropout -nss_dropout $nss_dropout -clf_dropout $clf_dropout > $log_path/$d.log &
# $d-$hops-$layers-$f-$lr-$l2_coef-$nas_dropout-$nss_dropout-$clf_dropout.log &


g=3
f=256
nrl=concat
hops=5
layers=1
lr=0.003
l2_coef=0.0
nas_dropout=0
nss_dropout=0.8
clf_dropout=0
b=200000
d=products s=ogb
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -nas_dropout $nas_dropout -nss_dropout $nss_dropout -clf_dropout $clf_dropout -b $b > $log_path/$d.log &
# $d-$hops-$layers-$f-$lr-$l2_coef-$nas_dropout-$nss_dropout-$clf_dropout-$b.log &


g=7
f=256
nrl=concat
hops=3
layers=1
lr=0.001
l2_coef=0.0
nas_dropout=0
nss_dropout=0.8
clf_dropout=0
b=200000
d=pokec s=linkx
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -nas_dropout $nas_dropout -nss_dropout $nss_dropout -clf_dropout $clf_dropout -b $b > $log_path/$d.log &
# $d-$hops-$layers-$f-$lr-$l2_coef-$nas_dropout-$nss_dropout-$clf_dropout-$b.log &
