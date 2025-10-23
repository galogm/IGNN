# RTX 3090, with Python 3.8.16, PyTorch 2.1.2, and Cuda 12.1

python -u -m main --gpu_id 0 --seed 42 --dataset actor --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 128 --lr 0.005 --l2_coef 5e-05 --n_hops 1 --n_layers 5 --early_stop 150 --RN attentive --norm_type none --act_type none --att_act_type relu --pre_dropout 0.5 --hid_dropout 0.6 --clf_dropout 0.0

python -u -m main --gpu_id 0 --seed 42 --dataset chameleon --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 128 --lr 0.005 --l2_coef 0.0005 --n_hops 1 --n_layers 5 --early_stop 200 --RN attentive --norm_type ln --act_type relu --att_act_type prelu --pre_dropout 0.8 --hid_dropout 0.7 --clf_dropout 0.4

python -u -m main --gpu_id 5 --seed 42 --dataset squirrel --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 64 --lr 0.005 --l2_coef 0.0005 --n_hops 1 --n_layers 3 --early_stop 200 --RN attentive --norm_type none --act_type prelu --att_act_type prelu --pre_dropout 0.8 --hid_dropout 0.4 --clf_dropout 0.1

python -u -m main --gpu_id 1 --seed 42 --dataset blogcatalog --source cola --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.001 --l2_coef 0.0001 --n_hops 1 --n_layers 3 --early_stop 200 --RN attentive --norm_type ln --act_type prelu --att_act_type tanh --pre_dropout 0.7 --hid_dropout 0.4 --clf_dropout 0.9

python -u -m main --gpu_id 1 --seed 42 --dataset flickr --source cola --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 1e-05 --n_hops 8 --n_layers 1 --early_stop 150 --RN attentive --norm_type none --act_type prelu --att_act_type leakyrelu --pre_dropout 0.8 --hid_dropout 0.2 --clf_dropout 0.8

python -u -m main --gpu_id 2 --seed 42 --dataset roman-empire --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.005 --l2_coef 5e-05 --n_hops 1 --n_layers 5 --early_stop 150 --RN attentive --norm_type ln --act_type prelu --att_act_type tanh --pre_dropout 0.4 --hid_dropout 0.2 --clf_dropout 0.3

python -u -m main --gpu_id 2 --seed 42 --dataset amazon-ratings --source critical --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 0.0 --n_hops 1 --n_layers 5 --early_stop 200 --RN attentive --norm_type none --act_type prelu --att_act_type gelu --pre_dropout 0.2 --hid_dropout 0.5 --clf_dropout 0.3

python -u -m main --gpu_id 3 --seed 42 --dataset photo --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 1e-05 --n_hops 1 --n_layers 3 --early_stop 200 --RN attentive --norm_type none --act_type prelu --att_act_type gelu --pre_dropout 0.5 --hid_dropout 0.8 --clf_dropout 0.9

python -u -m main --gpu_id 3 --seed 42 --dataset pubmed --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.005 --l2_coef 0.0001 --n_hops 1 --n_layers 3 --early_stop 50 --RN attentive --norm_type none --act_type prelu --att_act_type none --pre_dropout 0.3 --hid_dropout 0.4 --clf_dropout 0.3

python -u -m main --gpu_id 4 --seed 42 --dataset wikics --source pyg --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.001 --l2_coef 0.0005 --n_hops 10 --n_layers 1 --early_stop 200 --RN attentive --norm_type ln --act_type prelu --att_act_type none --pre_dropout 0.3 --hid_dropout 0.9 --clf_dropout 0.7

python -u -m main --gpu_id 0 --seed 42 --dataset arxiv --source ogb --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 512 --lr 0.0005 --l2_coef 1e-5 --n_hops 1 --n_layers 6 --early_stop 200 --RN attentive --norm_type ln --act_type relu --att_act_type sigmoid --pre_dropout 0.0 --hid_dropout 0.8 --clf_dropout 0.8

# 77.70±0.47
nohup python -u -m main --gpu_id 1 --seed 42 --dataset products --source ogb --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 200 --lr 0.001 --l2_coef 0.0 --n_hops 6 --n_layers 1 --early_stop 200 --RN attentive --norm_type ln --act_type relu --att_act_type leakyrelu --fast False --pre_dropout 0.2 --hid_dropout 0.4 --clf_dropout 0.2 --eval_start 280 -i 1  > logs/a-IGNN/pro.log &

nohup python -u -m main --gpu_id 2 --seed 42 --dataset products --source ogb --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 200 --lr 0.0005 --l2_coef 0.000001 --n_hops 6 --n_layers 1 --early_stop 200 --RN attentive --norm_type ln --act_type prelu --att_act_type sigmoid --fast False --pre_dropout 0.2 --hid_dropout 0.5 --clf_dropout 0.5 --eval_start 280 -i 3  > logs/a-IGNN/pro.log &

# 80.40±0.27
nohup python -u -m main --gpu_id 5 --seed 42 --dataset pokec --source linkx --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 200 --lr 0.001 --l2_coef 0.00001 --n_hops 1 --n_layers 5 --early_stop 200 --RN attentive --norm_type ln --act_type relu --att_act_type prelu --fast False --pre_dropout 0.1 --hid_dropout 0 --clf_dropout 0 --eval_start 1200 -i 5  > logs/a-IGNN/pokec.log &

nohup python -u -m main --gpu_id 5 --seed 42 --dataset pokec --source linkx --model ignn --n_epochs 3000 --agg_type gcn --IN IN-nSN --h_feats 256 --lr 0.0005 --l2_coef 0.000001 --n_hops 1 --n_layers 5 --early_stop 200 --RN attentive --norm_type ln --act_type relu --att_act_type prelu --fast False --pre_dropout 0 --hid_dropout 0.2 --clf_dropout 0.2 --eval_start 1200 -i 5  > logs/a-IGNN/pokec-1.log &
