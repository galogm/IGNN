hops=64 g=0 f=512 d=all s=pyg nie=gcn-nie-nst m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=32 g=7 f=512 d=all s=pyg nie=gcn-nie-nst m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=16 g=6 f=512 d=all s=pyg nie=gcn-nie-nst m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=8 g=5 f=512 d=all s=pyg nie=gcn-nie-nst m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=4 g=4 f=512 d=all s=pyg nie=gcn-nie-nst m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=2 g=3 f=512 d=all s=pyg nie=gcn-nie-nst m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=1 g=2 f=512 d=all s=pyg nie=gcn-nie-nst m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=0 g=1 f=512 d=all s=pyg nie=gcn-nie-nst m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

# or
g=5 f=512 d=all s=critical nie=deepsets nrl=concat m=FlatGNN-$nie-$nrl v=88.1 b=1024; nohup python -u main_hops.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/$m-$d-$s-$v-$nie-$nrl-hop-$nrl.log &

g=1 f=512 d=all s=critical nie=gcn-nie-nst nrl=concat m=FlatGNN-$nie-$nrl v=88.1 b=1024; nohup python -u main_hops.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/$m-$d-$s-$v-$nie-$nrl-hop-$nrl.log &

g=3 f=512 d=all s=critical nie=gcn-nie-nst nrl=lstm m=FlatGNN-$nie-$nrl v=88.1 b=1024; nohup python -u main_hops.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/$m-$d-$s-$v-$nie-$nrl-hop-$nrl.log &

g=4 f=512 d=all s=critical nie=gcn-nie-nst nrl=mean m=FlatGNN-$nie-$nrl v=88.1 b=1024; nohup python -u main_hops.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/$m-$d-$s-$v-$nie-$nrl-hop-$nrl.log &
