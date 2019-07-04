#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ..

#python eval_d.py --trained_model="./weights/voc_h/VOC.pth" --save_folder="./eval/voc_h/" --model_type="h"
weight_file=./weights/voc_h
eval_folder=./eval/voc_h
for((i=5000;i<120000;i+=5000));do
    weight=$weight_file/ssd300_COCO_$i.pth
    save_folder=$eval_folder/$i
    mkdir -p $save_folder
    echo $weight
    echo $save_folder
    python eval_d.py --trained_model=$weight --save_folder=$save_folder --model_type="h"
done