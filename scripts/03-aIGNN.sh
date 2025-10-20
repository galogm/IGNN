# RTX 3090, with Python 3.8.16, PyTorch 2.1.2, and Cuda 12.1

python -u -m main --gpu_id 0 --seed 42 --dataset actor --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 128 --lr 0.005 --l2_coef 5e-05 --n_hops 1 --n_layers 5 --early_stop 150 --RN attentive --norm_type none --act_type none --att_act_type relu --preln False --pre_dropout 0.5 --hid_dropout 0.6 --clf_dropout 0.0

python -u -m main --gpu_id 0 --seed 42 --dataset chameleon --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 128 --lr 0.005 --l2_coef 0.0005 --n_hops 1 --n_layers 5 --early_stop 200 --RN attentive --norm_type ln --act_type relu --att_act_type prelu --preln False --pre_dropout 0.8 --hid_dropout 0.7 --clf_dropout 0.4

python -u -m main --gpu_id 5 --seed 42 --dataset squirrel --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 64 --lr 0.005 --l2_coef 0.0005 --n_hops 1 --n_layers 3 --early_stop 200 --RN attentive --norm_type none --act_type prelu --att_act_type prelu --preln False --pre_dropout 0.8 --hid_dropout 0.4 --clf_dropout 0.1

python -u -m main --gpu_id 1 --seed 42 --dataset blogcatalog --source cola --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.001 --l2_coef 0.0001 --n_hops 1 --n_layers 3 --early_stop 200 --RN attentive --norm_type ln --act_type prelu --att_act_type tanh --preln True --pre_dropout 0.7 --hid_dropout 0.4 --clf_dropout 0.9

python -u -m main --gpu_id 1 --seed 42 --dataset flickr --source cola --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 1e-05 --n_hops 8 --n_layers 1 --early_stop 150 --RN attentive --norm_type none --act_type prelu --att_act_type leakyrelu --preln True --pre_dropout 0.8 --hid_dropout 0.2 --clf_dropout 0.8

python -u -m main --gpu_id 2 --seed 42 --dataset roman-empire --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.005 --l2_coef 5e-05 --n_hops 1 --n_layers 5 --early_stop 150 --RN attentive --norm_type ln --act_type prelu --att_act_type tanh --preln False --pre_dropout 0.4 --hid_dropout 0.2 --clf_dropout 0.3

python -u -m main --gpu_id 2 --seed 42 --dataset amazon-ratings --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 0.0 --n_hops 1 --n_layers 5 --early_stop 200 --RN attentive --norm_type none --act_type prelu --att_act_type gelu --preln False --pre_dropout 0.2 --hid_dropout 0.5 --clf_dropout 0.3

python -u -m main --gpu_id 3 --seed 42 --dataset photo --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 1e-05 --n_hops 1 --n_layers 3 --early_stop 200 --RN attentive --norm_type none --act_type prelu --att_act_type gelu --preln False --pre_dropout 0.5 --hid_dropout 0.8 --clf_dropout 0.9

python -u -m main --gpu_id 3 --seed 42 --dataset pubmed --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.005 --l2_coef 0.0001 --n_hops 1 --n_layers 3 --early_stop 50 --RN attentive --norm_type none --act_type prelu --att_act_type none --preln True --pre_dropout 0.3 --hid_dropout 0.4 --clf_dropout 0.3

python -u -m main --gpu_id 4 --seed 42 --dataset wikics --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 0.0005 --n_hops 10 --n_layers 1 --early_stop 200 --RN attentive --norm_type ln --act_type prelu --att_act_type none --preln False --pre_dropout 0.3 --hid_dropout 0.9 --clf_dropout 0.7

v=1.0
agg=gcn
IN=IN-nSN
RN=attentive
log_path=logs/$RN
mkdir -p $log_path
mkdir -p $log_path/large

# 72.57
python -u -m main --gpu_id 0 --seed 42 --dataset arxiv --source ogb --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 5e-5 --n_hops 1 --n_layers 5 --early_stop 200 --RN attentive --norm_type ln --act_type relu --att_act_type sigmoid --pre_dropout 0.2 --hid_dropout 0.8 --clf_dropout 0.5

nohup python -u -m main --gpu_id 1 --seed 42 --dataset products --source ogb --model ignn --n_epochs 300 --agg_type gcn --IN IN-nSN --h_feats 200 --lr 0.001 --l2_coef 0.0 --n_hops 1 --n_layers 5 --early_stop 100 --RN attentive --norm_type ln --act_type relu --att_act_type leakyrelu --preln True --fast False --pre_dropout 0.2 --hid_dropout 0.5 --clf_dropout 0.5 --eval_start 280 -i 1 --repeat 1  > logs/a-IGNN/pro.log &

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
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -pre_dropout $pre_dropout -hid_dropout $hid_dropout -clf_dropout $clf_dropout -a=$agg >$log_path/large/$d-$m.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "
# $d-$hops-$layers-$f-$lr-$l2_coef-$pre_dropout-$hid_dropout-$clf_dropout.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=products s=ogb
g=5
f=200
hops=5
layers=1
lr=0.003
l2_coef=0.00000
pre_dropout=0
hid_dropout=0.5
clf_dropout=0.5
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -pre_dropout $pre_dropout -hid_dropout $hid_dropout -clf_dropout $clf_dropout -eval 100 -i 1 -a=$agg >$log_path/large/$d-$hops-$layers-$f-$lr-$l2_coef-$pre_dropout-$hid_dropout-$clf_dropout-$b.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=pokec s=linkx
g=6
f=256
hops=5
layers=1
lr=0.001
l2_coef=0.000005
pre_dropout=0
hid_dropout=0.2
clf_dropout=0.2
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -pre_dropout $pre_dropout -hid_dropout $hid_dropout -clf_dropout $clf_dropout -eval 1200 -i 1 -a=$agg >$log_path/large/$d-$hops-$layers-$f-$lr-$l2_coef-$pre_dropout-$hid_dropout-$clf_dropout-$b.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "
