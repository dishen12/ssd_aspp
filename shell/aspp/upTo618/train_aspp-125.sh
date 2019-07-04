#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../..
python train_aspp.py --rate="1,2,5" --save_folder="./weights/voc_aspp_125/"