d=${1:-"cora"}
s=${2:-"pyg"}

public=${3:-"True"}
repeat=${4:-"5"}

RN=none
log_path=logs/dml/
mkdir -p $log_path

v=1.0
IN=nIN-nSN
g=0
f=512
layers=1
lr=0.001
l2_coef=0.00000
pre_dropout=0
hid_dropout=0.5
clf_dropout=0.5
m=GCN
agg_type=gcn

nohup bash -c "
for hops in 3 8 10 16 32 64;
do
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 python -u -m scripts.d_M_L \\
        -g \"$g\" -f \"$f\" -d \"$d\" -s \"$s\" -m \"$m\" -v \"$v\" \\
        -IN \"$IN\"  -RN \"$RN\" -hops \"\$hops\" -layers \"$layers\" \\
        -lr \"$lr\" -l2_coef \"$l2_coef\" \\
        --pre_dropout \"$pre_dropout\" \\
        --hid_dropout \"$hid_dropout\" \\
        --clf_dropout \"$clf_dropout\" \\
        --agg_type \"$agg_type\" -n ln \\
        --public \"$public\" --repeat \"$repeat\"
done
" > $log_path/GCN-$d.log 2>&1 & echo $!






RN=concat
log_path=logs/dml/
mkdir -p $log_path

v=1.0
IN=IN-SN
g=1
f=512
layers=1
lr=0.001
l2_coef=0.00000
pre_dropout=0
hid_dropout=0.5
clf_dropout=0.5
m=c-IGNN
agg_type=gcn_incep

nohup bash -c "
for hops in 3 8 10 16 32 64;
do
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 python -u -m scripts.d_M_L \\
        -g \"$g\" -f \"$f\" -d \"$d\" -s \"$s\" -m \"$m\" -v \"$v\" \\
        -IN \"$IN\"  -RN \"$RN\" -hops \"\$hops\" -layers \"$layers\" \\
        -lr \"$lr\" -l2_coef \"$l2_coef\" \\
        --pre_dropout \"$pre_dropout\" \\
        --hid_dropout \"$hid_dropout\" \\
        --clf_dropout \"$clf_dropout\" \\
        --agg_type \"$agg_type\" -n ln \\
        --public \"$public\" --repeat \"$repeat\"
done
" > $log_path/c-IGNN-$d.log 2>&1 & echo $!




RN=residual
log_path=logs/dml/
mkdir -p $log_path

v=1.0
IN=IN-nSN
g=2
f=512
layers=1
lr=0.001
l2_coef=0.00000
pre_dropout=0
hid_dropout=0.5
clf_dropout=0.5
m=r-IGNN
agg_type=gcn

nohup bash -c "
for hops in 3 8 10 16 32 64;
do
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 python -u -m scripts.d_M_L \\
        -g \"$g\" -f \"$f\" -d \"$d\" -s \"$s\" -m \"$m\" -v \"$v\" \\
        -IN \"$IN\"  -RN \"$RN\" -hops \"\$hops\" -layers \"$layers\" \\
        -lr \"$lr\" -l2_coef \"$l2_coef\" \\
        --pre_dropout \"$pre_dropout\" \\
        --hid_dropout \"$hid_dropout\" \\
        --clf_dropout \"$clf_dropout\" \\
        --agg_type \"$agg_type\" -n ln \\
        --public \"$public\" --repeat \"$repeat\"
done
" > $log_path/r-IGNN-$d.log 2>&1 & echo $!
