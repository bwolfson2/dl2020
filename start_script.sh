#!/bin/bash
apt install unzip #install unzip
apt install wget #install wget

#download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

#launch miniconda
bash ~/miniconda.sh -b -p $HOME/miniconda

#set miniconda activation
source /root/miniconda/bin/activate
conda init

#create gpu env
conda create -n gpu python=3.7 -y
conda activate gpu

#install packages
conda install pandas numpy pytorch torchvision opencv ipykernel jupyter ipython matplotlib  -y
conda install pillow=6.2.1 -y

#set to jupyter environment
python -m ipykernel install --user --name=gpu 
git clone https://github.com/bwolfson2/dl2020.git
cd dl2020

#download gcloud sdk
wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-290.0.0-linux-x86_64.tar.gz
tar -xvf google-cloud-sdk-290.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh --path-update true -q

#Open new terminal 
mv client.zip2 client.zip
unzip client.zip


#download student data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fCYFNpLopbUDOc5Pv3Gv1VD6GCVb5ash' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fCYFNpLopbUDOc5Pv3Gv1VD6GCVb5ash" -O student_data.zip && rm -rf /tmp/cookies.txt
unzip student_data.zip
rm student_data.zip

