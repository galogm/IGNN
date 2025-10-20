# Tesla V100, with Python 3.9.15, PyTorch 2.0.1, and Cuda 11.7

python -u -m main --gpu_id 0 --seed 42 --dataset actor --source pyg --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 512 --lr 0.001 --l2_coef 0.0 --n_hops 1 --n_layers 1 --early_stop 100 --RN concat --norm_type ln --act_type relu --preln False --fast False --pre_dropout 0.0 --hid_dropout 0.8 --clf_dropout 0.9

python -u -m main --gpu_id 3 --seed 42 --dataset chameleon --source critical --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 64 --lr 0.001 --l2_coef 0.0 --n_hops 1 --n_layers 5 --early_stop 150 --RN concat --norm_type none --act_type none --preln True --fast False --pre_dropout 0.8 --hid_dropout 0.3 --clf_dropout 0.3

python -u -m main --gpu_id 1 --seed 42 --dataset squirrel --source critical --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 128 --lr 0.005 --l2_coef 0.0 --n_hops 1 --n_layers 3 --early_stop 200 --RN none --norm_type none --act_type relu --preln False --fast False --pre_dropout 0.8 --hid_dropout 0.2 --clf_dropout 0.8

python -u -m main --gpu_id 3 --seed 42 --dataset flickr --source cola --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 128 --lr 0.001 --l2_coef 0.0001 --n_hops 8 --n_layers 1 --early_stop 100 --RN none --norm_type ln --act_type relu --preln False --fast True --pre_dropout 0.8 --hid_dropout 0.6 --clf_dropout 0.5

python -u -m main --gpu_id 3 --seed 42 --dataset blogcatalog --source cola --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 256 --lr 0.001 --l2_coef 1e-05 --n_hops 1 --n_layers 1 --early_stop 150 --RN none --norm_type none --act_type relu --preln True --fast False --pre_dropout 0.7 --hid_dropout 0.9 --clf_dropout 0.3

python -u -m main --gpu_id 4 --seed 42 --dataset roman-empire --source critical --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 256 --lr 0.01 --l2_coef 5e-05 --n_hops 1 --n_layers 5 --early_stop 200 --RN none --norm_type bn --act_type relu --preln True --fast False --pre_dropout 0.5 --hid_dropout 0.2 --clf_dropout 0.4

python -u -m main --gpu_id 4 --seed 42 --dataset amazon-ratings --source critical --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 300 --lr 0.001 --l2_coef 5e-05 --n_hops 16 --n_layers 1 --early_stop 300 --RN concat --norm_type ln --act_type prelu --preln False --fast False --pre_dropout 0.0 --hid_dropout 0.8 --clf_dropout 0.9

python -u -m main --gpu_id 1 --seed 42 --dataset photo --source pyg --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 512 --lr 0.001 --l2_coef 0.0 --n_hops 3 --n_layers 1 --early_stop 150 --RN none --norm_type bn --act_type relu --preln False --fast False --pre_dropout 0.5 --hid_dropout 0.6 --clf_dropout 0.2

python -u -m main --gpu_id 2 --seed 42 --dataset pubmed --source pyg --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 128 --lr 0.01 --l2_coef 0.0005 --n_hops 6 --n_layers 1 --early_stop 100 --RN concat --norm_type none --act_type relu --preln False --fast True --pre_dropout 0.2 --hid_dropout 0.5 --clf_dropout 0.6

python -u -m main --gpu_id 2 --seed 42 --dataset wikics --source pyg --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 512 --lr 0.005 --l2_coef 1e-05 --n_hops 6 --n_layers 1 --early_stop 150 --RN concat --norm_type ln --act_type prelu --preln True --fast True --pre_dropout 0.2 --hid_dropout 0.7 --clf_dropout 0.2

python -u -m main --gpu_id 2 --seed 42 --dataset arxiv --source ogb --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 512 --lr 0.001 --l2_coef 0.00005 --n_hops 8 --n_layers 1 --early_stop 200 --RN concat --norm_type bn --act_type relu --preln True --fast False --pre_dropout 0.0 --hid_dropout 0.8 --clf_dropout 0.5

python -u -m main --gpu_id 5 --seed 42 --dataset products --source ogb --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 200 --lr 0.0005 --l2_coef 0.0 --n_hops 1 --n_layers 8 --early_stop 100 --RN concat --norm_type ln --act_type relu --preln True --fast False --pre_dropout 0.0 --hid_dropout 0.5 --clf_dropout 0.5 --eval_start 280 -i 1 --repeat 1

nohup python -u -m main --gpu_id 3 --seed 42 --dataset pokec --source linkx --model ignn --n_epochs 2000 --agg_type gcn_incep --IN IN-SN --h_feats 256 --lr 0.001 --l2_coef 0.00005 --n_hops 1 --n_layers 5 --early_stop 100 --RN concat --norm_type bn --act_type relu --preln True --fast False --pre_dropout 0.0 --hid_dropout 0.2 --clf_dropout 0.2 --eval_start 1200 -i 5 > logs/c-IGNN/pokec.log 2>&1 &


v=1.0
agg=gcn_incep
IN=IN-SN
RN=concat
fs=True
log_path=logs/$RN
mkdir -p $log_path
mkdir -p $log_path/large

nohup python -u -m main --gpu_id 0 --seed 42 --dataset products --source ogb --model ignn --n_epochs 300 --agg_type gcn_incep --IN IN-SN --h_feats 200 --lr 0.001 --l2_coef 0.0 --n_hops 1 --n_layers 5 --early_stop 100 --RN concat --norm_type ln --act_type relu --preln True --fast False --pre_dropout 0.0 --hid_dropout 0.5 --clf_dropout 0.5 --eval_start 280 -i 1 --repeat 1 > logs/c-IGNN/pro.log &

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


nohup python -u -m main --gpu_id 5 --seed 42 --dataset products --source ogb --model ignn --n_epochs 2000 --agg_type gcn_incep --IN IN-SN --h_feats 256 --lr 0.001 --l2_coef 0.0000005 --n_hops 1 --n_layers 5 --early_stop 100 --RN concat --norm_type ln --act_type relu --preln True --fast False --pre_dropout 0.0 --hid_dropout 0.2 --clf_dropout 0.2 --eval_start 1200 -i 5 --repeat 1 > logs/c-IGNN/pok.log &

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
