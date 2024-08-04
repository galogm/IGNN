hops=32 g=7 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=16 g=6 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=8 g=5 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=4 g=4 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=2 g=3 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=1 g=2 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &

hops=0 g=1 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$hops v=99.8 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie -hops $hops > logs/$m-$d-$s-$v.log &
