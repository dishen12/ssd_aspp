#!/bin/bash
source /home/luban/.bashrc
source /etc/profile
source /home/luban/miniconda3/bin/activate base
cd ..
#eval_folder=./eval/voc_i
#$i=5000
#for file in ./weights/voc_i/*.pth;do 
    #echo $file
    #save_folder=$eval_folder/$i
    #mkdir -p $save_folder
    #echo $save_folder
    #i=$[$i+5000];
    #python eval_d.py --trained_model=$file --save_folder=$save_folder --model_type="i"
#done


#python eval_d.py --trained_model="./weights/voc_i/VOC.pth" --save_folder="./eval/voc_i/" --model_type="i"

weight_file=./weights/voc_i
eval_folder=./eval/voc_i
for((i=5000;i<120000;i+=5000));do
    weight=$weight_file/ssd300_COCO_$i.pth
    save_folder=$eval_folder/$i
    mkdir -p $save_folder
    echo $weight
    echo $save_folder
    python eval_d.py --trained_model=$weight --save_folder=$save_folder --model_type="i"
done