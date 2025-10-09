mkdir -p logs

GPU=0
d=roman-empire
s=critical
r=10
nohup python -u -m main.py --dataset $d --source $s --hidden_channels 64 --local_epochs 100 --global_epochs 2500 --lr 0.001 --runs $r --local_layers 10 --global_layers 2 --weight_decay 0.0 --dropout 0.3 --global_dropout 0.5 --in_dropout 0.15 --num_heads 8 --device $GPU --save_model --beta 0.5 > logs/$d.log &

GPU=1
d=amazon-ratings
s=critical
r=10
nohup python -u -m main --dataset $d --source $s --hidden_channels 256 --local_epochs 200 --global_epochs 2500 --lr 0.001 --runs $r --local_layers 10 --global_layers 1 --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 2 --device $GPU  > logs/$d.log &


GPU=0
d=wikics
s=pyg
r=10
nohup python -u -m main.py --dataset $d --source $s --hidden_channels 512 --local_epochs 100 --global_epochs 1000 --lr 0.001 --runs $r --local_layers 7 --global_layers 2 --weight_decay 0.0 --dropout 0.5 --in_dropout 0.5 --num_heads 1 --device $GPU --save_model > logs/$d.log &
