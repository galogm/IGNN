f=512 v=1.0
nie=gcn-nie-nst

g=0
nrl=concat
hops=None
d=actor s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/concat/$d.log &

g=0
nrl=concat
hops=None
d=chameleon s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/concat/$d.log &

g=6
nrl=concat
hops=None
d=squirrel s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/concat/$d.log &

g=2
nrl=concat
hops=None
d=flickr s=cola
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/concat/$d.log &

g=2
nrl=concat
hops=None
d=blogcatalog s=cola
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/concat/$d.log &

g=3
nrl=concat
hops=None
d=roman-empire s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/concat/$d.log &

g=7
nrl=concat
hops=None
d=amazon-ratings s=critical
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/concat/$d.log &

g=4
nrl=concat
hops=None
d=photo s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/concat/$d.log &

g=5
nrl=concat
hops=None
d=pubmed s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/concat/$d.log &

g=5
nrl=concat
hops=None
d=wikics s=pyg
m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > logs/concat/$d.log &
