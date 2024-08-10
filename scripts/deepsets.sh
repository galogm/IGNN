f=512 nie=deepsets v=99.1 b=1024;

d=actor s=pyg;
g=0;
nrl=None m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
d=chameleon s=critical
g=0;
nrl=None m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
d=squirrel s=critical
g=1;
nrl=None m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
d=blogcatalog s=cola
g=1;
nrl=None m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
d=flickr s=cola
g=2;
nrl=None m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
d=photo s=pyg
g=2;
nrl=None m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
d=roman-empire s=critical
g=3;
nrl=None m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
d=amazon-ratings s=critical
g=4;
nrl=None m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
d=pubmed s=pyg
g=5;
nrl=None m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
d=wikics s=pyg
g=3;
nrl=None m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &


# d=chameleon s=critical
# d=flickr s=cola
# d=pubmed s=pyg
# d=photo s=pyg
# d=wikics s=pyg

# g=5;
# nrl=sum m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
# g=6;
# nrl=lstm m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
# g=7;
# nrl=mean m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
# g=8;
# nrl=max m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
# g=0;
# nrl=concat m=FlatGNN-$nie-$nrl; nohup python -u main.py -g $g -f $f -d $d -s $s -m $m -v $v -b $b -nie $nie  -nrl $nrl  > logs/deepsets-$d-$nrl.log &
