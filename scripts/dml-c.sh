RN=concat
log_path=logs/dml/$RN
mkdir -p $log_path

v=1.0
IN=gcn-IN-SN
g=2
f=512
layers=1
lr=0.001
l2_coef=0.00000
# nas_dropout=0
# nss_dropout=0.5
# clf_dropout=0.5
# d=cora s=pyg
# m=IGNN-$IN-$RN

# for hops in 3 8 10 16 32 64 80;
# do
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 python -u d_M_L.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef -nas_dropout $nas_dropout -nss_dropout $nss_dropout -clf_dropout $clf_dropout --eval_start 0;
# done


d=squirrel s=critical
m=IGNN-$IN-$RN

for hops in 3 8 10 16 32 64 80;
do
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 python -u d_M_L.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops -layers $layers -lr $lr -l2_coef $l2_coef --eval_start 0;
done
