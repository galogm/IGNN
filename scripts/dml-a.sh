RN=attentive
log_path=logs/dml/$RN
mkdir -p $log_path

v=1.0
IN=gcn
g=0
f=512
hops=10
layers=1
lr=0.001
l2_coef=0.00000
pre_dropout=0
hid_dropout=0.8
clf_dropout=0.8
d=cora s=pyg
m=IGNN-$IN-$RN; PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 nohup python -u d_M_L.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -pre_dropout $pre_dropout -hid_dropout $hid_dropout -clf_dropout $clf_dropout --eval_start 0 -at sigmoid > $log_path/$d-$hops-$layers-$f-$lr-$l2_coef-$pre_dropout-$hid_dropout-$clf_dropout-$b.log & echo $!
