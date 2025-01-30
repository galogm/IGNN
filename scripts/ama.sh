# nohup bash scripts/ama.sh > logs/ama-88.log &
f=512 v=88.1
nie=gcn
nrl=residual
g=5
d=amazon-ratings
s=critical

for hop in 64 32 16 10 8 4 2 1; do
    m=FlatGNN-$nie-$nrl; python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hop
done
