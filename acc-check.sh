#!/bin/bash


models=(
  "mobilenet_v3_small"
  "mobilenet_v3_large"
  "efficientnet_b1"
  "efficientnet_b3"
  "densenet169"
  "densenet201"
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
      --batch-size 8
      --path models/${model}-${dataset}-${i}.pth
    )

    python3 -m benchmark.acc-check "${args[@]}"
  done
done
