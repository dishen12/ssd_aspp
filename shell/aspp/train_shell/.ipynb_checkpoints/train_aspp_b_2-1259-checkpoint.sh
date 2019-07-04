#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../..
python train_aspp.py --model_type="aspp_b_2" --rate="1,2,5,9" --save_folder="./weights/voc_aspp_b_2_1259_extra_rate/"
# extra rate  modify to the same rate as rate ,there has some error there 
# attention , there has bn before !!!!!!!!!!