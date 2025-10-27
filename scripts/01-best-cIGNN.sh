# Tesla V100, with Python 3.9.15, PyTorch 2.0.1, and Cuda 11.7
python -u -m main --gpu_id 0 --seed 42 --dataset actor --source pyg --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 512 --lr 0.001 --l2_coef 0.0 --n_hops 1 --n_layers 1 --early_stop 100 --RN concat --norm_type ln --act_type relu --preln False --fast False --pre_dropout 0.0 --hid_dropout 0.8 --clf_dropout 0.9

python -u -m main --gpu_id 0 --seed 42 --dataset chameleon --source critical --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 64 --lr 0.001 --l2_coef 0.0 --n_hops 1 --n_layers 5 --early_stop 150 --RN concat --norm_type none --act_type none --preln True --fast False --pre_dropout 0.8 --hid_dropout 0.3 --clf_dropout 0.3

python -u -m main --gpu_id 0 --seed 42 --dataset squirrel --source critical --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 128 --lr 0.005 --l2_coef 0.0 --n_hops 1 --n_layers 3 --early_stop 200 --RN none --norm_type none --act_type relu --preln False --fast False --pre_dropout 0.8 --hid_dropout 0.2 --clf_dropout 0.8

python -u -m main --gpu_id 0 --seed 42 --dataset flickr --source cola --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 128 --lr 0.001 --l2_coef 0.0001 --n_hops 8 --n_layers 1 --early_stop 100 --RN none --norm_type ln --act_type relu --preln False --fast True --pre_dropout 0.8 --hid_dropout 0.6 --clf_dropout 0.5

python -u -m main --gpu_id 0 --seed 42 --dataset blogcatalog --source cola --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 256 --lr 0.001 --l2_coef 1e-05 --n_hops 1 --n_layers 1 --early_stop 150 --RN none --norm_type none --act_type relu --preln True --fast False --pre_dropout 0.7 --hid_dropout 0.9 --clf_dropout 0.3

python -u -m main --gpu_id 0 --seed 42 --dataset roman-empire --source critical --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 256 --lr 0.01 --l2_coef 5e-05 --n_hops 1 --n_layers 5 --early_stop 200 --RN none --norm_type bn --act_type relu --preln True --fast False --pre_dropout 0.5 --hid_dropout 0.2 --clf_dropout 0.4

python -u -m main --gpu_id 0 --seed 42 --dataset amazon-ratings --source critical --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 300 --lr 0.001 --l2_coef 5e-05 --n_hops 16 --n_layers 1 --early_stop 300 --RN concat --norm_type ln --act_type prelu --preln False --fast False --pre_dropout 0.0 --hid_dropout 0.8 --clf_dropout 0.9

python -u -m main --gpu_id 0 --seed 42 --dataset photo --source pyg --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 512 --lr 0.001 --l2_coef 0.0 --n_hops 3 --n_layers 1 --early_stop 150 --RN none --norm_type bn --act_type relu --preln False --fast False --pre_dropout 0.5 --hid_dropout 0.6 --clf_dropout 0.2

python -u -m main --gpu_id 0 --seed 42 --dataset pubmed --source pyg --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 128 --lr 0.01 --l2_coef 0.0005 --n_hops 6 --n_layers 1 --early_stop 100 --RN concat --norm_type none --act_type relu --preln False --fast True --pre_dropout 0.2 --hid_dropout 0.5 --clf_dropout 0.6

python -u -m main --gpu_id 0 --seed 42 --dataset wikics --source pyg --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 512 --lr 0.005 --l2_coef 1e-05 --n_hops 6 --n_layers 1 --early_stop 150 --RN concat --norm_type ln --act_type prelu --preln True --fast True --pre_dropout 0.2 --hid_dropout 0.7 --clf_dropout 0.2

# RTX 3090, with Python 3.8.16, PyTorch 2.1.2, and Cuda 12.1
python -u -m main --gpu_id 3 --seed 42 --dataset arxiv --source ogb --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 512 --lr 0.001 --l2_coef 0.00005 --n_hops 6 --n_layers 1 --early_stop 200 --RN concat --norm_type bn --act_type relu --preln True --fast False --pre_dropout 0.0 --hid_dropout 0.8 --clf_dropout 0.5 --public True --repeat 3

python -u -m main --gpu_id 0 --seed 42 --dataset products --source ogb --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 200 --lr 0.0005 --l2_coef 0.0 --n_hops 1 --n_layers 5 --early_stop 100 --RN concat --norm_type ln --act_type relu --preln True --fast False --pre_dropout 0.0 --hid_dropout 0.5 --clf_dropout 0.5 --eval_start 280 -i 1 -public True --repeat 3

python -u -m main --gpu_id 0 --seed 42 --dataset pokec --source linkx --model ignn --n_epochs 3000 --agg_type gcn_incep --IN IN-SN --h_feats 256 --lr 0.001 --l2_coef 0.00005 --n_hops 1 --n_layers 5 --early_stop 100 --RN concat --norm_type bn --act_type relu --preln True --fast False --pre_dropout 0.0 --hid_dropout 0.2 --clf_dropout 0.2 --eval_start 1200 -i 5 --public True --repeat 5
