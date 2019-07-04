#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../..
python train_aspp.py --model_type="aspp_b_2_mutilRate" --rate="1,2,5,9,6,3,2,1,2,3,5,7" --save_folder="./weights/voc_aspp_b_2_1259_mutilRate/"