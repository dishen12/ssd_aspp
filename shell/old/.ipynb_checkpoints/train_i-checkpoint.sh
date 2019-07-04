#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ..
python train_d.py --model_type="i" --save_folder="./weights/voc_i/"