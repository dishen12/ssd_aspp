#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ../..

weight_file=./weights/voc_aspp_a_1_1259
eval_folder=./eval/voc_aspp_a_1_1259
for((i=190000;i<=200000;i+=5000));do
    weight=$weight_file/ssd300_VOC_$i.pth
    save_folder=$eval_folder/$i
    mkdir -p $save_folder
    echo $weight
    echo $save_folder
    python eval_aspp.py --model_type="aspp_a_1" --trained_model=$weight --save_folder=$save_folder --rate="1,2,5,9"
done
#python eval_aspp.py --trained_model="./weights/voc_aspp_1259/VOC.pth" --save_folder="./eval/voc_aspp-1259/"
