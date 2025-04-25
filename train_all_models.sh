#!/bin/bash

models=(
"resnet18_pretrained" "resnet18"
"resnet50_pretrained" "resnet50"
"AlexNet_pretrained" "AlexNet"
"ConvNeXt_pretrained" "ConvNeXt"
"vit_b_16_pretrained" "vit_b_16"
"vit_l_16_pretrained" "vit_l_16"
"swin_b_pretrained" "swin_b"
"swin_v2_b_pretrained" "swin_v2_b"
"densenet121_pretrained" "densenet121"
"resnext50_32x4d_pretrained" "resnext50_32x4d"
"mnasnet1_0_pretrained" "mnasnet1_0"
"shufflenet_v2_x1_0_pretrained" "shufflenet_v2_x1_0"
"vgg16_pretrained" "vgg16"
"vgg19_pretrained" "vgg19"
)

batch_size=128
lr=0.001
num_epochs=20
dataset="tiny-imagenet"
logdir="results"
num_workers=16
save_checkpoint_every=5
seed=42

mkdir -p saved_models

for model in "${models[@]}"
do
    echo "Training $model ..."
    python train.py \
        --model $model \
        --batch_size $batch_size \
        --lr $lr \
        --num_epochs $num_epochs \
        --parallel \
        --dataset $dataset \
        --logdir $logdir \
        --num_workers $num_workers \
        --save_checkpoint_every $save_checkpoint_every \
        --seed $seed
    echo "Finished $model."
done