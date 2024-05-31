#!/bin/bash


models=(
  # "mobilenet_v3_small"
  # "mobilenet_v3_large"
  # "efficientnet_b1"
  # "efficientnet_b3"
  # "densenet169"
  # "densenet201"
  "vit_b_16"
)

dataset="Flowers102"

for model in "${models[@]}"
do
  for i in {1..5}
  do
    args=(
      --model $model
      --dataset $dataset
      --epochs 10
      --batch-size 8
      --save-path models/${model}-${dataset}-${i}.pth
    )

    python3 train.py "${args[@]}"
  done
done
