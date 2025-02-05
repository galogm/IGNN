log_path=logs/residual
mkdir -p $log_path

f=512 v=1.0
nie=gcn

g=0
nrl=residual
hops=2
d=actor s=pyg
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=0
nrl=residual
hops=1
d=chameleon s=critical
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=6
nrl=residual
hops=2
d=squirrel s=critical
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=6
nrl=residual
hops=2
d=flickr s=cola
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=4
nrl=residual
hops=4
d=blogcatalog s=cola
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=3
nrl=residual
hops=2
d=roman-empire s=critical
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=7
nrl=residual
hops=2
d=amazon-ratings s=critical
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=4
nrl=residual
hops=4
d=photo s=pyg
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=6
nrl=residual
hops=2
d=pubmed s=pyg
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &

g=0
nrl=residual
hops=4
d=wikics s=pyg
m=IGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -nie $nie  -nrl $nrl -hops $hops > $log_path/$d.log &
