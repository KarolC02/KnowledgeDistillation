#!/bin/bash

# teachers=("ConvNeXt_pretrained" "resnet50_pretrained" "resnext50_32x4d_pretrained" "densenet121_pretrained")
# students=("shufflenet_v2_x1_0_pretrained" "mnasnet1_0_pretrained" "resnet18_pretrained" "resnet50_pretrained")

teachers=("ConvNeXt_pretrained")
students=("shufflenet_v2_x1_0_pretrained")

dataset="tiny-imagenet"
logdir="results"
logits_dir="saved_logits"
modeldir="saved_models"
teacher_checkpoint_convnext="checkpoint_epoch_5.pth"
teacher_checkpoint_default="final_checkpoint.pth"

for teacher in "${teachers[@]}"
do
    for student in "${students[@]}"
    do
        echo "Distilling from $teacher to $student..."

        checkpoint=$teacher_checkpoint_default
        if [[ "$teacher" == "ConvNeXt_pretrained" ]]; then
            checkpoint=$teacher_checkpoint_convnext
        fi

        python distill.py \
            --teacher_model $teacher \
            --student_model $student \
            --dataset $dataset \
            --logdir $logdir \
            --logits_dir $logits_dir \
            --modeldir $modeldir \
            --parallel \
            --adapt_model \
            --teacher_checkpoint_name $checkpoint
    done
done
