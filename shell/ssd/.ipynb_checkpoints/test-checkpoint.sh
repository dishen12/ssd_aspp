#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../../

python eval.py --trained_model="./weights/voc/VOC.pth" --save_folder="./eval/voc/"