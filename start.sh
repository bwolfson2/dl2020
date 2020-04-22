# !/bin/bash
apt install unzip
apt install wget
wget https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
bash Miniconda3-py37_4.8.2-Linux-x86_64.sh
mkdir DL_conda
cd DL_conda
conda create-n gpu python=3.7
conda install pandas numpy tensorflow-gpu keras opencv ipykernel
python -m ipykernel install --user --name=gpu 
source /root/miniconda/bin/activate
conda init