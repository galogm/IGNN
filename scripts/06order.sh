f=512 v=99.16 b=1024
nie=gcn-nie-nst

g=0
nrl=ordered-gating
hops=None
d=actor s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/ab/$d-$nrl-0813.log &

g=0
nrl=ordered-gating
hops=None
d=chameleon s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/ab/$d-$nrl-0813.log &

g=1
nrl=ordered-gating
hops=None
d=squirrel s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/ab/$d-$nrl-0813.log &

g=2
nrl=ordered-gating
hops=None
d=flickr s=cola
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/ab/$d-$nrl-0813.log &

g=3
nrl=ordered-gating
hops=None
d=blogcatalog s=cola
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/ab/$d-$nrl-0813.log &

g=4
nrl=ordered-gating
hops=None
d=roman-empire s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f 256 -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/ab/$d-$nrl-0813.log &

g=0
nrl=ordered-gating
hops=None
d=amazon-ratings s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f 256 -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/ab/$d-$nrl-0813.log &

g=1
nrl=ordered-gating
hops=None
d=photo s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/ab/$d-$nrl-0813.log &

g=5
nrl=ordered-gating
hops=None
d=pubmed s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f 256 -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/ab/$d-$nrl-0813.log &

g=3
nrl=ordered-gating
hops=None
d=wikics s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f 256 -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl -hops $hops > logs/ab/$d-$nrl-0813.log &
