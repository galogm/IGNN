# nohup bash scripts/ch.sh > logs/chameleon-88.log &
f=512 v=88.1
nie=gcn
nrl=residual
g=0
d=chameleon
s=critical

for hop in 1 2 4 8 10 16 24 32 64; do
    m=FlatGNN-$nie-$nrl; python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hop
done
