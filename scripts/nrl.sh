nrl=lstm g=4 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$nrl v=99.9 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl > logs/$m-$d-$s-$v.log &

nrl=sum g=5 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$nrl v=99.9 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl > logs/$m-$d-$s-$v.log &

nrl=mean g=5 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$nrl v=99.9 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl > logs/$m-$d-$s-$v.log &

nrl=max g=7 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$nrl v=99.9 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl > logs/$m-$d-$s-$v.log &

nrl=concat g=7 f=512 d=all s=pyg nie=gcn m=FlatGNN-$nie-$nrl v=99.9 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl > logs/$m-$d-$s-$v.log &
