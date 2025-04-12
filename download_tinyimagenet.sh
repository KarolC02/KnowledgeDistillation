#!/bin/bash
# Download the tinyimagenet dataset,unzip it and move it to the datasets directory

if [ -d "datasets/tiny-imagenet-200" ]; then
    echo "Tiny ImagenNet already present in datasets directory. Skipping Download."
    exit 0
fi

echo "Tiny ImageNet not found. Downloading..."

mkdir -p datasets

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -O datasets/tiny-imagenet-200.zip
unzip datasets/tiny-imagenet-200.zip -d datasets/

rm datasets/tiny-imagenet-200.zip

echo "Tiny ImageNet downloaded!"