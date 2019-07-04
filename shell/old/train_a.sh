#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ..
python train_d.py --model_type="a" --save_folder="./weights/voc_a/" --lr=1e-4