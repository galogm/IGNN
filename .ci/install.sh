# # install python 3.8 use pyenv
# USE_SSH=true curl https://pyenv.run | bash
# sudo apt-get install zlib1g-dev libffi-dev libreadline-dev libssl-dev libsqlite3-dev libncurses5 libncurses5-dev libncursesw5 lzma liblzma-dev libbz2-dev
# pyenv install 3.8
# pyenv local 3.8

# create and activate virtual environment
if [ ! -d '.env' ]; then
    python3 -m venv .env && echo create venv
else
    echo venv exists
fi

source .env/bin/activate

# # update pip
python3 -m pip install -U pip

# # torch cuda
python3 -m pip install "torch==2.1.2" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  -c constraints.txt
python3 -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html  -c constraints.txt

# # dgl cuda
python3 -m pip install dgl==2.0 -f https://data.dgl.ai/wheels/cu121/repo.html  -c constraints.txt

# # install requirements
python3 -m pip install -r requirements.txt  -c constraints.txt

echo install requirements successfully!
