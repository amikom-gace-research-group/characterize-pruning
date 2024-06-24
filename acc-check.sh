#!/bin/bash


models=(
  # "mobilenet_v3_small"
  # "mobilenet_v3_large"
  # "efficientnet_b1"
  # "efficientnet_b3"
  "densenet169"
  "densenet201"
  "vit_b_16"
)

dataset="Flowers102"

for model in "${models[@]}"
do
    sl="sparse_bsc"
    path="models/pruned/0.8-RND-${model}-Flowers102-${sl}.pth"
    args=(
      --model $model
      --dataset $dataset
      --batch-size 8
      --path "${path}"
    )

    python3 -m benchmark.acc-check "${args[@]}"
done
