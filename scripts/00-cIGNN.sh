log_path=logs/concat
mkdir -p $log_path

f=512 v=1.0
nie=gcn-nie-nst

g=0
nrl=concat
hops=1
d=actor s=pyg
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=0
nrl=concat
hops=64
d=chameleon s=critical
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=6
nrl=concat
hops=64
d=squirrel s=critical
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=2
nrl=concat
hops=10
d=flickr s=cola
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=2
nrl=concat
hops=10
d=blogcatalog s=cola
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=3
nrl=concat
hops=1
d=roman-empire s=critical
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=7
nrl=concat
hops=10
d=amazon-ratings s=critical
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=4
nrl=concat
hops=10
d=photo s=pyg
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=5
nrl=concat
hops=4
d=pubmed s=pyg
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=5
nrl=concat
hops=8
d=wikics s=pyg
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &
