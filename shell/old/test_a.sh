#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ..

#python eval_d.py --trained_model="./weights/voc_a/ssd300_COCO_115000.pth" --save_folder="./eval/voc_a/" --model_type="a"

weight_file=./weights/voc_a
eval_folder=./eval/voc_a
for((i=5000;i<120000;i+=5000));do
    weight=$weight_file/ssd300_COCO_$i.pth
    save_folder=$eval_folder/$i
    mkdir -p $save_folder
    echo $weight
    echo $save_folder
    python eval_d.py --trained_model=$weight --save_folder=$save_folder --model_type="a"
done