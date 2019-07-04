#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../..
python train_aspp.py --model_type="aspp_b_2_left_last" --rate="1,2,5,9" --save_folder="./weights/voc_aspp_b_2_left_last_1259/"