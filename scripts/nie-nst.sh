g=6 f=512 d=all s=critical nie=gcn-nnie-st nrl=none m=FlatGNN-$nie-$nrl v=99.6 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl > logs/$m-$d-$s-$v-$nie-$nrl.log &

g=6 f=512 d=all s=critical nie=gcn-nnie-st nrl=concat m=FlatGNN-$nie-$nrl v=99.6 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl > logs/$m-$d-$s-$v-$nie-$nrl.log &

g=6 f=512 d=all s=critical nie=gcn-nie-st nrl=concat m=FlatGNN-$nie-$nrl v=99.6 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl > logs/$m-$d-$s-$v-$nie-$nrl.log &

g=6 f=512 d=all s=critical nie=gcn-nnie-nst nrl=concat m=FlatGNN-$nie-$nrl v=99.6 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl > logs/$m-$d-$s-$v-$nie-$nrl.log &

g=6 f=512 d=all s=critical nie=gcn-nie-nst nrl=concat m=FlatGNN-$nie-$nrl v=99.6 b=1024; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl > logs/$m-$d-$s-$v-$nie-$nrl.log &
