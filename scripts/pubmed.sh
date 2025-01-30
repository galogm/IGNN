# nohup bash scripts/pubmed.sh > logs/pubmed-88.log &
f=512 v=88.1
nie=gcn
nrl=residual
g=7
d=pubmed
s=pyg

for hop in 1 2 4 8 10 16 32 64; do
    m=FlatGNN-$nie-$nrl; python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hop
done
