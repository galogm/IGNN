log_path=logs/concat
mkdir -p $log_path

f=512 v=1.0
IN=gcn-IN-SN

g=0
RN=concat
hops=1
d=actor s=pyg
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=0
RN=concat
hops=64
d=chameleon s=critical
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=1
RN=concat
hops=64
d=squirrel s=critical
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=1
RN=concat
hops=10
d=flickr s=cola
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=0
RN=concat
hops=10
d=blogcatalog s=cola
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=0
RN=concat
hops=1
d=roman-empire s=critical
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=1
RN=concat
hops=10
d=amazon-ratings s=critical
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=0
RN=concat
hops=10
d=photo s=pyg
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=1
RN=concat
hops=4
d=pubmed s=pyg
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &

g=1
RN=concat
hops=8
d=wikics s=pyg
m=IGNN-$IN-$RN; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -IN $IN  -RN $RN -hops $hops > $log_path/$d.log &
