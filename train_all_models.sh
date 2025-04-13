#!/bin/bash

# Optionally activate your environment
# source /path/to/your/venv/bin/activate

# List of models
models=("resnet18_pretrained" "resnet50_pretrained" "AlexNet_pretrained" "ConvNeXt_pretrained" "myCNN" "fastCNN" "OneLayerNN")

# Fixed training parameters
batch_size=128
lr=0.001
num_epochs=1
parallel=true

# Make a directory to save models
mkdir -p saved_models

for model in "${models[@]}"
do
    echo "Training $model ..."

    python train.py \
        --model $model \
        --batch_size $batch_size \
        --lr $lr \
        --num_epochs $num_epochs \
        --parallel $parallel

    echo "Saving $model ..."
    cp "logs/tiny_image_net_${model}_lr=${lr}_epochs=${num_epochs}_batch_size=${batch_size}/checkpoint.pth" "saved_models/${model}.pth"
done

echo "🏌️‍♂️ All models trained and saved. Go enjoy your golf!"
