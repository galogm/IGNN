# g=1 nrl=self-attention;nohup bash scripts/00sota-seq.sh $g $nrl >logs/00sota-seq.log &
f=512 v=99.1
g=$1
nrl=$2

nie=gcn-nie-nst
m=FlatGNN-$nie-$nrl
# b=None
hops=None

# 定义每个数组
critical=("chameleon" "squirrel" "roman-empire" "amazon-ratings")
cola=("flickr" "blogcatalog")
pyg=("photo" "actor" "pubmed" "wikics")
# pyg=('wikics')

# 定义一个关联数组来存储这些数组的名称
declare -A DATASETS

# 将数组通过名称关联到关联数组中
DATASETS[critical]="critical"
DATASETS[cola]="cola"
DATASETS[pyg]="pyg"
DATASETS[Critical]="Critical"

# 函数：根据item查找对应的key
function find_key_by_item() {
    local item="$1"

    for key in "${!DATASETS[@]}"; do
        # 获取当前key对应的数组名称
        array_name="${DATASETS[$key]}"
        # 使用 eval 和间接引用访问数组元素
        eval "arr=(\"\${${array_name}[@]}\")"
        for value in "${arr[@]}"; do
            if [[ "$value" == "$item" ]]; then
                echo "$key"
                return
            fi
        done
    done
    echo "Item not found"
}

# 遍历数据集并运行命令
for d in "actor" "blogcatalog" "flickr" "roman-empire" "squirrel" "chameleon" "amazon-ratings" "pubmed" "photo" "wikics"; do
    echo "Processing dataset: $d"
    source=$(find_key_by_item "$d")
    if [[ "$source" != "Item not found" ]]; then
        echo "Source for $d: $source"
        python -u main.py -g $g -f $f -d $d -s $source -m $m  -v $v -nie $nie  -nrl $nrl -hops $hops
    else
        echo "Dataset $d not found in any source."
    fi
done
