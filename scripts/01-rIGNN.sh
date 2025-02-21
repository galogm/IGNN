log_path=logs/residual
mkdir -p $log_path

f=512 v=1.0
IN=gcn

g=0
RN=residual
hops=2
d=actor s=pyg
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=0
RN=residual
hops=1
d=chameleon s=critical
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=6
RN=residual
hops=2
d=squirrel s=critical
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=6
RN=residual
hops=2
d=flickr s=cola
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=4
RN=residual
hops=4
d=blogcatalog s=cola
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=3
RN=residual
hops=2
d=roman-empire s=critical
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=7
RN=residual
hops=2
d=amazon-ratings s=critical
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=4
RN=residual
hops=4
d=photo s=pyg
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=6
RN=residual
hops=2
d=pubmed s=pyg
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=0
RN=residual
hops=4
d=wikics s=pyg
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &
