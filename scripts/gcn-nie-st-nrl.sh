f=512 v=99.8 b=1024;

g=0
nie=gcn-nie-st
nrl=None
hops=None
d=actor s=pyg;
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/st-$d-$nrl-0811.log &
g=1
d=chameleon s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/st-$d-$nrl-0811.log &
g=2
d=squirrel s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/st-$d-$nrl-0811.log &

g=3;
nie=gcn-nie-st
nrl=None
hops=None
d=blogcatalog s=cola
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/st-$d-$nrl-0811.log &
g=5
d=flickr s=cola
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/st-$d-$nrl-0811.log &

# g=1;
# nie=gcn-nie-st
# nrl=None
# hops=None
# d=roman-empire s=critical
# m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/st-$d-$nrl-0811.log &
# # d=amazon-ratings s=critical
# # m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/st-$d-$nrl-0811.log &

# g=2;
# nie=gcn-nie-st
# nrl=None
# hops=None
# d=pubmed s=pyg
# m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/st-$d-$nrl-0811.log &

# g=3;
# nie=gcn-nie-st
# nrl=None
# hops=None
# d=wikics s=pyg
# m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/st-$d-$nrl-0811.log &

# g=4;
# nie=gcn-nie-st
# nrl=None
# hops=None
# d=photo s=pyg
# m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/st-$d-$nrl-0811.log &
