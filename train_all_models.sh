#!/bin/bash

# List of models
models=("resnet18_pretrained" "resnet50_pretrained" "AlexNet_pretrained" "ConvNeXt_pretrained" "myCNN" "fastCNN" "OneLayerNN")
# models=("myCNN" "fastCNN" "OneLayerNN")

batch_size=128
lr=0.001
num_epochs=10

mkdir -p saved_models

for model in "${models[@]}"
do
    echo "Training $model ..."

    if [[ "$model" == myCNN || "$model" == fastCNN || "$model" == OneLayerNN || "$model" == myImprovedCNN ]]; then
        python train.py \
            --model $model \
            --batch_size $batch_size \
            --lr $lr \
            --num_epochs $num_epochs 
    else 
        python train.py \
            --model $model \
            --batch_size $batch_size \
            --lr $lr \
            --num_epochs $num_epochs \
            --parallel
    fi
    echo "Saving $model ..."
    cp "logs/tiny_image_net_${model}_lr=${lr}_epochs=${num_epochs}_batch_size=${batch_size}/checkpoint.pth" "saved_models/${model}.pth"
done