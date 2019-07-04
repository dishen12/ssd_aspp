#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../..
#old
#python train_aspp.py --model_type="aspp_a" --rate="1,2,5,9" --save_folder="./weights/voc_aspp_a_1259/"
#again
python train_aspp.py --model_type="aspp_a" --rate="1,2,5,9" --save_folder="./weights/voc_aspp_a_1259_again/"