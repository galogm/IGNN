f=512 nie=gcn-nie-nst v=99.1 b=1024;

# d=actor s=pyg;
# d=chameleon s=critical
# d=squirrel s=critical
# d=blogcatalog s=cola
# d=flickr s=cola
# d=roman-empire s=critical
# d=amazon-ratings s=critical
# d=pubmed s=pyg
# d=photo s=pyg
d=wikics s=pyg


g=0;
nrl=sum m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &
g=1;
nrl=lstm m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &
g=2;
nrl=mean m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &
g=3;
nrl=max m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &
g=4;
nrl=concat m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &
g=5;
nrl=only-concat m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &


# d=chameleon s=critical
# d=flickr s=cola
# d=pubmed s=pyg
# d=photo s=pyg
# d=wikics s=pyg

# g=5;
# nrl=sum m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &
# g=6;
# nrl=lstm m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &
# g=7;
# nrl=mean m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &
# g=8;
# nrl=max m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &
# g=0;
# nrl=concat m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/ablation-$d-$nrl.log &
