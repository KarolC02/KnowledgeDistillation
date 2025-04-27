#!/bin/bash

teachers=("ConvNeXt_pretrained")
students=("resnet18_pretrained")

dataset="tiny-imagenet"
logdir="results"
logits_dir="saved_logits"
modeldir="saved_models"
teacher_checkpoint_convnext="checkpoint_epoch_5.pth"
teacher_checkpoint_default="final_checkpoint.pth"

temperatures=(1 2 4 10 20 100)
alphas=(0 0.1 0.5 0.9 1)

for teacher in "${teachers[@]}"; do
    for student in "${students[@]}"; do
        for T in "${temperatures[@]}"; do
            for alpha in "${alphas[@]}"; do
                echo "Distilling from $teacher to $student with T=$T and alpha=$alpha..."

                checkpoint=$teacher_checkpoint_default
                if [[ "$teacher" == "ConvNeXt_pretrained" ]]; then
                    checkpoint=$teacher_checkpoint_convnext
                fi

                python DK.py \
                    --teacher_model $teacher \
                    --student_model $student \
                    --dataset $dataset \
                    --logdir $logdir \
                    --logits_dir $logits_dir \
                    --modeldir $modeldir \
                    --parallel \
                    --adapt_model \
                    --teacher_checkpoint_name $checkpoint \
                    --temperature $T \
                    --alpha $alpha
            done
        done
    done
done
