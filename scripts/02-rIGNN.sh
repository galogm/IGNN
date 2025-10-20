# Tesla V100, with Python 3.9.15, PyTorch 2.0.1, and Cuda 11.7

python -u -m main --gpu_id 5 --seed 42 --dataset actor --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 0.0005 --n_hops 1 --n_layers 3 --early_stop 100 --RN residual --norm_type none --act_type relu --preln True --pre_dropout 0.3 --hid_dropout 0.5 --clf_dropout 0.5

python -u -m main --gpu_id 5 --seed 42 --dataset chameleon --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 128 --lr 0.01 --l2_coef 5e-05 --n_hops 1 --n_layers 3 --early_stop 50 --RN residual --norm_type ln --act_type relu --preln True --pre_dropout 0.3 --hid_dropout 0.6 --clf_dropout 0.9

python -u -m main --gpu_id 5 --seed 42 --dataset squirrel --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 128 --lr 0.005 --l2_coef 5e-05 --n_hops 1 --n_layers 3 --early_stop 200 --RN residual --norm_type none --act_type relu --preln False --pre_dropout 0.8 --hid_dropout 0.8 --clf_dropout 0.8

python -u -m main --gpu_id 6 --seed 42 --dataset flickr --source cola --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.001 --l2_coef 0.0 --n_hops 3 --n_layers 1 --early_stop 50 --RN residual --norm_type ln --act_type none --preln True --pre_dropout 0.6 --hid_dropout 0.9 --clf_dropout 0.9

python -u -m main --gpu_id 2 --seed 42 --dataset blogcatalog --source cola --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.005 --l2_coef 5e-05 --n_hops 1 --n_layers 3 --early_stop 150 --RN residual --norm_type bn --act_type prelu --preln False --pre_dropout 0.2 --hid_dropout 0.1 --clf_dropout 0.2

python -u -m main --gpu_id 7 --seed 42 --dataset roman-empire --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.005 --l2_coef 0.0005 --n_hops 1 --n_layers 5 --early_stop 200 --RN residual --norm_type bn --act_type relu --preln False --pre_dropout 0.4 --hid_dropout 0.7 --clf_dropout 0.3

python -u -m main --gpu_id 3 --seed 42 --dataset amazon-ratings --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 5e-05 --n_hops 1 --n_layers 3 --early_stop 150 --RN residual --norm_type bn --act_type relu --preln True --pre_dropout 0.2 --hid_dropout 0.8 --clf_dropout 0.2

python -u -m main --gpu_id 2 --seed 42 --dataset photo --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.001 --l2_coef 0.0005 --n_hops 1 --n_layers 3 --early_stop 200 --RN residual --norm_type ln --act_type prelu --preln True --pre_dropout 0.5 --hid_dropout 0.6 --clf_dropout 0.5

python -u -m main --gpu_id 7 --seed 42 --dataset pubmed --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.001 --l2_coef 1e-05 --n_hops 1 --n_layers 3 --early_stop 50 --RN residual --norm_type ln --act_type prelu --preln True --pre_dropout 0.5 --hid_dropout 0.1 --clf_dropout 0.3

python -u -m main --gpu_id 5 --seed 42 --dataset wikics --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.005 --l2_coef 0.0005 --n_hops 3 --n_layers 1 --early_stop 150 --RN residual --norm_type ln --act_type prelu --preln True --pre_dropout 0.4 --hid_dropout 0.1 --clf_dropout 0.3


# 72.69
python -u -m main --gpu_id 1 --seed 42 --dataset arxiv --source ogb --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.001 --l2_coef 0.00001 --n_hops 1 --n_layers 5 --early_stop 200 --RN residual --norm_type ln --act_type prelu --preln False --pre_dropout 0.2 --hid_dropout 0.5 --clf_dropout 0.8

python -u -m main --gpu_id 4 --seed 42 --dataset products --source ogb --model ignn --n_epochs 300 --agg_type gcn --IN IN-nSN --h_feats 200 --lr 0.003 --l2_coef 0.0 --n_hops 1 --n_layers 5 --early_stop 200 --RN residual --norm_type ln --act_type relu --fast False --pre_dropout 0.0 --hid_dropout 0.5 --clf_dropout 0.5 --eval_start 280 -i 1

python -u -m main --gpu_id 4 --seed 42 --dataset pokec --source ogb --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.001 --l2_coef 0.00001 --n_hops 5 --n_layers 1 --early_stop 100 --RN residual --norm_type bn --act_type relu --fast False --pre_dropout 0.0 --hid_dropout 0.2 --clf_dropout 0.2 --eval_start 1200 -i 1

v=1.0
agg=gcn
IN=IN-nSN
RN=residual
log_path=logs/$RN
mkdir -p $log_path
mkdir -p $log_path/large

d=products s=ogb
g=2
f=200
hops=5
layers=1
lr=0.003
l2_coef=0.00000
pre_dropout=0
hid_dropout=0.5
clf_dropout=0.5
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -pre_dropout $pre_dropout -hid_dropout $hid_dropout -clf_dropout $clf_dropout --eval_start 100 -i 3 -n ln -p 5 -a=$agg >$log_path/large/$d-$hops-$layers-$f-$lr-$l2_coef-$pre_dropout-$hid_dropout-$clf_dropout-$b.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "

d=pokec s=linkx
g=3
f=256
hops=5
layers=1
lr=0.001
l2_coef=0.000005
pre_dropout=0
hid_dropout=0.2
clf_dropout=0.2
m=IGNN-$IN-$RN
nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -pre_dropout $pre_dropout -hid_dropout $hid_dropout -clf_dropout $clf_dropout -eval 1200 -i 3  -n bn -a=$agg >$log_path/large/$d-$hops-$layers-$f-$lr-$l2_coef-$pre_dropout-$hid_dropout-$clf_dropout-$b.log 2>&1 & echo "Check logs in $log_path/$d-$m.log. PID: $! "
