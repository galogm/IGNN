# nohup bash scripts/sq.sh > logs/squirrel-88.log &
f=512 v=88.1
nie=gcn
nrl=residual
g=0
d=squirrel
s=critical

for hop in 64 32 16 10 8 4 2 1; do
    m=FlatGNN-$nie-$nrl; python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hop
done
